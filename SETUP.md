# Setup

Prepare the machines, then continue with [WORKFLOW.md](WORKFLOW.md). Recording
and training run on a workstation (Linux or Windows); inference runs on the edge
Linux PC and sends ArtNet to the MADRIX PC.

- `<edge-user>` — the edge service account, typically `sed`.
- `$SED_HOME` — the edge repo path, typically `/home/sed/SoundDetection2`.
- Edge OS: Lubuntu or Ubuntu Server 24.04/26.04 (Lubuntu for local GUI tools,
  Ubuntu Server for headless).

## Workstation (Linux)

### Install packages

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git curl build-essential python3 python3-venv python3-pip \
    rsync openssh-client libsndfile1 libportaudio2 libasound2t64 alsa-utils pciutils
```

Add `openssh-server` only if other machines must connect into this workstation.

### Install uv and the repo

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
git clone <repo> SoundDetection2 && cd SoundDetection2
git submodule update --init --recursive
uv sync --extra train --extra dev
uv run sed --help
uv run pytest
```

### Check audio and GPU

Plug in the USB audio interface first.

```bash
arecord -l
uv run sed audio devices
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is `False`, training still runs on CPU, just slower. To install GPU
drivers: `ubuntu-drivers devices`, then `sudo ubuntu-drivers autoinstall && sudo reboot`.

## Workstation (Windows 10/11)

In PowerShell as Administrator:

```powershell
winget install --id Git.Git -e
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reopen PowerShell, then:

```powershell
git clone <repo> SoundDetection2; cd SoundDetection2
git submodule update --init --recursive
uv sync --extra train --extra dev
uv run sed audio devices
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Install the audio interface driver if Windows does not detect it. WSL2 is
optional but not the main path for USB audio.

## Edge Device

The production Linux PC: SSH-reachable, sees the USB interface, runs the `sed`
service. It does not need the training environment or the shared library.

Install choices: hostname `sed-edge`, one admin user (`<edge-user>`), enable
OpenSSH, disable sleep/suspend, keep on AC power.

### Install packages

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git curl build-essential python3 python3-venv python3-pip \
    openssh-server rsync avahi-daemon network-manager \
    libsndfile1 libportaudio2 libasound2t64 alsa-utils libusb-1.0-0
sudo systemctl enable --now ssh avahi-daemon NetworkManager
```

### Clone and install the service

Clone into the service account's home, so the repo is `$SED_HOME`
(`/home/sed/SoundDetection2`). It runs **in place** — nothing is copied to `/opt`.

```bash
cd ~
git clone <repo> SoundDetection2 && cd SoundDetection2
git submodule update --init --recursive
sudo PROJECT_ID=<project_id> bash ./deploy/install.sh
```

Use any placeholder `<project_id>` if the final one is unknown. The installer runs
`uv sync` (inference deps only, no torch) and installs/enables the systemd units:

- `sed.service` — the 24/7 inference service.
- `sed-model.path` — watches config/model changes and restarts the service.
- `sed-model-reload.service` — the restart helper.
- `/etc/sed/env` — holds the selected `PROJECT_ID`.

`sed.service` is enabled but not started until a model is deployed (see
[WORKFLOW.md](WORKFLOW.md#deploy-to-the-edge)). Re-run the installer after a
`git pull` to resync; it never touches `projects/`.

### Verify

```bash
arecord -l
uv run sed audio devices
systemctl is-enabled sed sed-model.path
systemctl status sed
```

It is OK if `sed.service` is inactive before a model exists.

## Network

### Same-Wi-Fi SSH

To control the edge from a laptop on the same Wi-Fi. On the edge:

```bash
sudo nmtui                       # join the Wi-Fi
sudo hostnamectl set-hostname sed-edge
sudo systemctl restart avahi-daemon
hostname -I
```

From another computer:

```bash
ssh <edge-user>@sed-edge.local   # or: ssh <edge-user>@<edge-ip>
```

If `ufw` is enabled on the edge: `sudo ufw allow OpenSSH`. Reserve the edge IP in
the router/DHCP so it does not change between sessions.

### Static ArtNet Ethernet

One wired network for ArtNet and on-site SSH.

| Device | IP |
|---|---|
| MADRIX Windows PC | `192.168.0.1/24` |
| Edge Linux PC | `192.168.0.100/24` |
| Optional workstation | `192.168.0.101/24` |

`192.168.0.1` is the MADRIX address, not a router — leave **gateway and DNS empty**
on every wired profile and keep internet on Wi-Fi.

- **Edge:** `sudo nmtui` → wired connection → IPv4 Manual → add `192.168.0.100/24`,
  no gateway/DNS → activate. Verify with `ip addr` and `ping 192.168.0.1`.
- **MADRIX Windows:** Settings → Network → the ArtNet Ethernet adapter → IPv4
  manual → `192.168.0.1`, mask `255.255.255.0`, no gateway/DNS. Verify with
  `ipconfig` and `ping 192.168.0.100`.

### Remote support via the MADRIX PC

Path: laptop → TeamViewer → MADRIX Windows PC → SSH → edge.

```powershell
ssh <edge-user>@192.168.0.100
```

Any computer on the network can SSH with the IP, the edge username, and the
password. To log in without a password, add your public key to the edge:

```powershell
ssh-keygen -t ed25519 -C "madrix-sed"
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

```bash
# on the edge
mkdir -p ~/.ssh && nano ~/.ssh/authorized_keys   # paste the public key
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

If password SSH was disabled, re-enable it: put `PasswordAuthentication yes` in
`/etc/ssh/sshd_config.d/99-sed.conf`, then `sudo systemctl restart ssh`.

## Reference

- **Ports:** SSH TCP 22, ArtNet UDP 6454 — they can share one subnet.
- **Audio OK** when `arecord -l` and `sed audio devices` show the USB interface
  with the expected inputs and a stable name substring (e.g. `UMC404HD`).
- **Sources:** Ubuntu
  [24.04](https://documentation.ubuntu.com/release-notes/24.04/) /
  [26.04](https://documentation.ubuntu.com/release-notes/26.04/), OnLogic
  [CL200 series](https://support.onlogic.com/product-documentation/industrial-products/cl200-series/cl250-cl210-cl200).

### Troubleshooting

- Cannot SSH to edge: IP from `hostname -I`, cable/Wi-Fi, `systemctl status ssh`,
  username/password.
- `sed-edge.local` won't resolve: use the IP directly; check `avahi-daemon`.
- No audio device: USB cable, `arecord -l`, `sed audio devices`, `libportaudio2`.
- NVIDIA unavailable: `nvidia-smi`, `ubuntu-drivers devices`.
- Service inactive after install: OK until a model bundle is deployed.
- No ping on ArtNet Ethernet: static IPs, subnet mask, wrong adapter, cable/switch.
