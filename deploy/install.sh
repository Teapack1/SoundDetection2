#!/usr/bin/env bash
# Idempotent IN-PLACE installer for the SED edge device. Safe to re-run.
# Tested on Lubuntu/Ubuntu 24.04 and 26.04 LTS; also handles 22.04 package names.
#
# The system runs from THIS cloned repo — nothing is copied to /opt. Clone the
# repo into the edge account's home (e.g. /home/sed/SoundDetection2), then:
#
#   sudo PROJECT_ID=my_venue bash ./deploy/install.sh
#
# A fresh clone has no projects/ (models are not uploaded to the cloud), so the
# installer sets up the environment, scaffolds projects/<id>/config.toml, and
# tells you to copy the trained project's `models/` folder + `config.toml` from
# the workstation into projects/<id>/. Re-run (or just start the service) once
# the bundle is in place.
#
# Optional overrides: SED_USER (default: the repo's owner, normally 'sed').

set -euo pipefail
trap 'echo "ERROR: install failed at line ${LINENO}. Nothing started; re-run after fixing." >&2' ERR

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID to the venue id, e.g. 'my_venue'}"
# The repo stays where it was cloned; the live system runs from here.
SED_HOME="$(cd "$(dirname "$0")/.." && pwd)"
# The service runs as whoever owns the repo (the edge has a single 'sed' user
# whose home holds the clone). Override with SED_USER=... if needed.
SED_USER="${SED_USER:-$(stat -c '%U' "$SED_HOME")}"

# --- preflight ---------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run as root (e.g. 'sudo PROJECT_ID=$PROJECT_ID bash $0')." >&2
    exit 1
fi
if [ ! -f "$SED_HOME/deploy/sed.service" ] || [ ! -f "$SED_HOME/pyproject.toml" ]; then
    echo "ERROR: run this from inside the cloned repo (missing deploy/ or pyproject.toml)." >&2
    exit 1
fi
if ! id -u "$SED_USER" >/dev/null 2>&1; then
    echo "ERROR: user '$SED_USER' does not exist. Create it (or set SED_USER) and" >&2
    echo "       clone the repo into its home, then re-run." >&2
    exit 1
fi
SED_USER_HOME="$(getent passwd "$SED_USER" | cut -d: -f6)"

retry() {  # retry <n> <cmd...> — for transient network/apt hiccups
    local n="$1"; shift; local i=1
    until "$@"; do
        if [ "$i" -ge "$n" ]; then return 1; fi
        echo "   (retry $i/$n: $*)"; sleep 3; i=$((i + 1))
    done
}

pick_pkg() {  # echo first apt-installable candidate (handles renamed packages)
    local p
    for p in "$@"; do
        if apt-cache show "$p" >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    echo "$1"  # fall back to the first; apt will report a clear error if missing
}

echo "==> 1) system packages"
export DEBIAN_FRONTEND=noninteractive
retry 3 apt-get update
# libasound2 was renamed to libasound2t64 on Ubuntu 24.04+ (64-bit time_t).
ASOUND_PKG="$(pick_pkg libasound2t64 libasound2)"
retry 3 apt-get install -y --no-install-recommends \
    python3 \
    libsndfile1 \
    libportaudio2 "$ASOUND_PKG" alsa-utils \
    libusb-1.0-0 \
    ca-certificates git rsync curl
# Always ensure audio group membership (idempotent; fixes pre-existing users too).
usermod -aG audio "$SED_USER"

echo "==> 2) uv (Python env manager — same tool used on the workstation)"
# Install uv system-wide once so both this script and the service-owner can use
# it. The edge uses the SAME 'uv run sed' workflow as the workstation; the
# difference is only which dependency groups are synced.
if ! command -v uv >/dev/null 2>&1; then
    retry 3 sh -c 'curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh'
fi
UV_BIN="$(command -v uv || echo /usr/local/bin/uv)"
echo "    uv: $("$UV_BIN" --version 2>/dev/null || echo 'not found')"

echo "==> 3) python venv (inference deps only — no torch on the edge)"
# `uv sync` (no extras) installs ONLY the inference dependencies declared in
# pyproject [project.dependencies] plus the `sed` console script — torch and the
# rest of the training stack live in the opt-in [train] extra and are skipped.
# It creates $SED_HOME/.venv, owned by $SED_USER.
run_as_user() { sudo -u "$SED_USER" -H env HOME="$SED_USER_HOME" "$@"; }
retry 2 run_as_user "$UV_BIN" sync --directory "$SED_HOME" --no-dev
SED_BIN="$SED_HOME/.venv/bin/sed"

echo "==> 4) /etc/sed/env"
mkdir -p /etc/sed
cat > /etc/sed/env <<EOF
PROJECT_ID=${PROJECT_ID}
PYTHONUNBUFFERED=1
EOF
chmod 0644 /etc/sed/env

echo "==> 5) logging + caches"
# Must exist before the service starts: sed.service lists these in ReadWritePaths=
# and systemd refuses to start a unit whose paths are missing. /var/cache/sed
# holds the numba (librosa) + matplotlib + openvino caches, which cannot be
# written inside the read-only repo/venv under ProtectSystem=strict.
mkdir -p /var/log/sed /var/cache/sed
chown "$SED_USER":"$SED_USER" /var/log/sed /var/cache/sed

echo "==> 6) project scaffold"
# Ensure projects/<id>/ exists with a user-editable config.toml. On a fresh
# clone there is no model yet — that is expected and handled below.
run_as_user "$SED_BIN" project new "$PROJECT_ID"
MODEL_ROOT="$SED_HOME/projects/${PROJECT_ID}/models"
mkdir -p "$MODEL_ROOT"
chown -R "$SED_USER":"$SED_USER" "$SED_HOME/projects/${PROJECT_ID}"

echo "==> 7) systemd units"
# Substitute the repo path + service user into the unit, and PROJECT_ID into the
# path-watcher (%i). The repo path varies per machine, so the units are templated.
sed_subst() { /usr/bin/sed -e "s|__SED_HOME__|${SED_HOME}|g" -e "s|__SED_USER__|${SED_USER}|g"; }
sed_subst < "$SED_HOME/deploy/sed.service" > /etc/systemd/system/sed.service
/usr/bin/sed -e "s|__SED_HOME__|${SED_HOME}|g" -e "s|%i|${PROJECT_ID}|g" \
    "$SED_HOME/deploy/sed-model.path" > /etc/systemd/system/sed-model.path
cp "$SED_HOME/deploy/sed-model-reload.service" /etc/systemd/system/sed-model-reload.service
systemctl daemon-reload
systemctl enable sed.service
systemctl enable --now sed-model.path

# A model is deployed if any models/<model_id>/ subfolder holds a config
# snapshot plus an OpenVINO IR.
model_deployed=0
for d in "$MODEL_ROOT"/*/; do
    [ -f "${d}training_config.json" ] || continue
    if [ -f "${d}model_int8.xml" ] || [ -f "${d}model_fp16.xml" ]; then
        model_deployed=1
        break
    fi
done
# Clear any prior failed state / start-limit from earlier runs so restart works.
systemctl reset-failed sed.service 2>/dev/null || true
if [ "$model_deployed" = 1 ]; then
    echo "==> starting sed.service (first start compiles numba kernels; ~seconds)"
    systemctl restart sed.service
    started=1
else
    echo "==> no model yet; sed.service is ENABLED (auto-starts on boot)"
    echo "    but NOT started — there is nothing to run until a model is copied in."
    started=0
fi

# --- final summary -----------------------------------------------------------
echo
echo "============================================================"
echo " INSTALL COMPLETE"
echo "============================================================"
echo
echo "The live system runs IN PLACE from $SED_HOME (owned by '$SED_USER')."
echo "The service runs:  sed inference --project $PROJECT_ID"
echo "It is ENABLED, so it starts automatically on every boot."
if [ "$started" = 1 ]; then
    echo "Status now: RUNNING (a model bundle was found under $MODEL_ROOT)."
else
    echo "Status now: IDLE (enabled, waiting for a model)."
    echo
    echo "NEXT STEP — copy the trained project from the workstation:"
    echo "  This is a fresh clone, so projects/$PROJECT_ID/ has no model yet."
    echo "  On the workstation, copy TWO things into projects/$PROJECT_ID/ here:"
    echo "    1. the whole  models/        folder   (the trained bundle(s))"
    echo "    2. the        config.toml    file     (venue config; source of truth)"
    echo "  e.g. from the workstation:"
    echo "    rsync -a projects/$PROJECT_ID/models/ \\"
    echo "        $SED_USER@<edge>:$SED_HOME/projects/$PROJECT_ID/models/"
    echo "    rsync -a projects/$PROJECT_ID/config.toml \\"
    echo "        $SED_USER@<edge>:$SED_HOME/projects/$PROJECT_ID/config.toml"
    echo "  Then pick the active model and start:"
    echo "    sudo -u $SED_USER $SED_BIN models select --project $PROJECT_ID <model_id>"
    echo "    sudo systemctl restart sed"
fi
echo
echo "Confirm it is running and will keep running:"
echo "  systemctl is-enabled sed     # -> enabled  (starts on boot)"
echo "  systemctl is-active sed      # -> active   (running now)"
echo "  systemctl status sed"
echo "  journalctl -u sed -f         # live logs; look for the READY line"
echo
echo "Common commands (identical to the workstation — uv run, or the venv binary):"
echo "  list models:   uv run --directory $SED_HOME sed models list --project $PROJECT_ID"
echo "  swap model:    sudo -u $SED_USER $SED_BIN models select --project $PROJECT_ID <model_id> && sudo systemctl restart sed"
echo
echo "Change runtime behaviour in projects/$PROJECT_ID/config.toml, then restart sed."
echo "The path watcher also restarts sed automatically when that config changes."
echo "Examples: [deploy] active_model/record_device/verbose, [mining] enabled, [artnet] enabled."
echo
echo "To update the code later: 'git pull' in $SED_HOME, then re-run this installer"
echo "(it re-syncs deps with uv and refreshes the systemd units). Your projects/"
echo "(models, config, recordings) are never touched by the installer."
