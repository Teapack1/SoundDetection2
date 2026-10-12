"""Sanity tests for the audio ring buffer."""

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Reads must never hand back a splice
# ---------------------------------------------------------------------------


def test_read_window_refuses_a_window_overwritten_during_the_copy():
    """The writer keeps running while a read copies.

    Checking only before the copy proves the window was intact at that instant
    and nothing more: a window close to the buffer's age gets overwritten
    underneath the reader, which used to return audio spliced from two
    different laps of the ring with no error at all. The window here is big
    enough that copying it takes long enough for the writer to catch up.
    """
    import threading

    from sed.audio.capture import RingBuffer

    capacity = 4_000_000
    ring = RingBuffer(capacity, channels=1)
    ring.write(np.zeros((capacity, 1), dtype="float32"))

    stop = threading.Event()
    chunk = np.ones((65536, 1), dtype="float32")

    def writer():
        while not stop.is_set():
            ring.write(chunk)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        outcomes = {"ok": 0, "raised": 0}
        for _ in range(40):
            start = ring.write_idx - (capacity - 200_000)
            try:
                ring.read_window(start, capacity - 400_000, channel=0)
                outcomes["ok"] += 1
            except ValueError:
                outcomes["raised"] += 1
        # Whatever the timing, a read either returns or raises — it must never
        # quietly return a spliced window.
        assert outcomes["raised"] > 0, outcomes
    finally:
        stop.set()
        t.join(timeout=2.0)


def test_read_window_refuses_a_window_spanning_a_dropout():
    from sed.audio.capture import RingBuffer

    ring = RingBuffer(4096, channels=1)
    ring.write(np.ones((1000, 1), dtype="float32"))
    ring.mark_discontinuity()
    ring.write(np.full((1000, 1), 2.0, dtype="float32"))

    # Wholly on one side: fine. Straddling the join: refused, because the two
    # halves are adjacent in the buffer but seconds apart in time.
    assert ring.read_window(0, 500, channel=0).shape == (500,)
    assert ring.read_window(1000, 500, channel=0).shape == (500,)
    with pytest.raises(ValueError, match="dropout"):
        ring.read_window(800, 400, channel=0)


def test_read_last_refuses_more_than_has_been_captured():
    from sed.audio.capture import RingBuffer

    ring = RingBuffer(4096, channels=1)
    ring.write(np.ones((100, 1), dtype="float32"))
    # Reaching back past the start of the stream used to wrap onto the
    # unwritten tail and return zeros as if they were audio.
    with pytest.raises(ValueError):
        ring.read_last(500, channel=0)
    assert np.all(ring.read_last(100, channel=0) == 1.0)


def test_write_refuses_a_block_larger_than_the_buffer():
    from sed.audio.capture import RingBuffer

    ring = RingBuffer(1024, channels=1)
    with pytest.raises(ValueError):
        ring.write(np.zeros((2048, 1), dtype="float32"))


def test_zero_length_reads_are_empty_not_errors():
    from sed.audio.capture import RingBuffer

    ring = RingBuffer(1024, channels=1)
    ring.write(np.ones((512, 1), dtype="float32"))
    assert ring.read_window(0, 0, channel=0).shape == (0,)
    assert ring.read_last(0, channel=0).shape == (0,)
