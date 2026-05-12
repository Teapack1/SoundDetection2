"""Sanity tests for the audio ring buffer."""

import numpy as np

from sed.audio.capture import RingBuffer


def test_ring_basic_write_and_read():
    rb = RingBuffer(capacity_samples=1000, channels=1)
    block = np.arange(300, dtype=np.float32)[:, None]
    rb.write(block)
    out = rb.read_last(300)
    assert np.array_equal(out, block[:, 0])


def test_ring_wraparound():
    rb = RingBuffer(capacity_samples=100, channels=1)
    for i in range(5):
        block = np.full((30, 1), i, dtype=np.float32)
        rb.write(block)
    last = rb.read_last(60)
    # last 60 samples come from the last two writes (i=3 and i=4 zones)
    assert last[0] == 3
    assert last[-1] == 4


def test_ring_window_read():
    rb = RingBuffer(capacity_samples=200, channels=1)
    rb.write(np.arange(150, dtype=np.float32)[:, None])
    win = rb.read_window(50, 50)
    assert np.array_equal(win, np.arange(50, 100, dtype=np.float32))
