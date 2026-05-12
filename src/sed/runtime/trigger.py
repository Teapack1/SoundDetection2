"""Publish trigger events over ZeroMQ for downstream consumers.

The detection process always publishes events on a local IPC socket; the
ArtNet "show controller" subscribes on the same socket. This decouples
detection (must stay reliable) from show logic (gets edited often).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

import zmq

from sed.runtime.smooth import TriggerEvent

log = logging.getLogger(__name__)


class EventPublisher:
    def __init__(self, endpoint: str = "ipc:///tmp/sed-events.sock"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.endpoint = endpoint
        try:
            self.sock.bind(endpoint)
        except zmq.ZMQError as e:
            log.warning("bind %s failed (%s) — falling back to tcp://*:5556", endpoint, e)
            self.sock.bind("tcp://*:5556")
        # Slow joiner: ZMQ PUB drops messages if no SUB has connected yet.
        # Brief sleep gives subscribers a chance to attach.
        time.sleep(0.2)
        log.info("publishing on %s", endpoint)

    def publish(self, event: TriggerEvent) -> None:
        msg = json.dumps({**asdict(event)}).encode("utf-8")
        try:
            self.sock.send(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        try:
            self.sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass


class EventSubscriber:
    """For the show controller (or anything else) to receive events."""

    def __init__(self, endpoint: str = "ipc:///tmp/sed-events.sock"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(endpoint)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")

    def recv(self, timeout_ms: int = 100) -> dict | None:
        if self.sock.poll(timeout=timeout_ms):
            return json.loads(self.sock.recv())
        return None

    def close(self) -> None:
        try:
            self.sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
