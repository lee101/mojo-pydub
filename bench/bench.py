"""Benchmarks mojo-pydub against upstream pydub on identical PCM buffers."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from mojo_pydub import AudioSegment  # noqa: E402
from pydub import AudioSegment as PyAudioSegment  # noqa: E402


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def timeit(function, repeat=5) -> float:
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def segments(seconds: int, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.integers(
        -28000,
        28001,
        size=seconds * 44100 * channels,
        dtype=np.int16,
    )
    kwargs = {"sample_width": 2, "frame_rate": 44100, "channels": channels}
    data = values.tobytes()
    return AudioSegment(data, **kwargs), PyAudioSegment(data, **kwargs)


def cases():
    minute, py_minute = segments(60)
    half, py_half = segments(30, seed=1)
    ten, py_ten = segments(10)
    return [
        (
            "apply_gain, 60 s stereo",
            lambda: minute.apply_gain(-3.0),
            lambda: py_minute.apply_gain(-3.0),
        ),
        (
            "normalize, 60 s stereo",
            lambda: minute.normalize(),
            lambda: py_minute.normalize(),
        ),
        (
            "overlay, 60 s + 30 s stereo",
            lambda: minute.overlay(half, position=10000),
            lambda: py_minute.overlay(py_half, position=10000),
        ),
        (
            "fade_in, 10 s stereo",
            lambda: ten.fade_in(8000),
            lambda: py_ten.fade_in(8000),
        ),
        (
            "pan, 60 s stereo",
            lambda: minute.pan(0.35),
            lambda: py_minute.pan(0.35),
        ),
        (
            "low_pass_filter, 10 s stereo",
            lambda: ten.low_pass_filter(1200),
            lambda: py_ten.low_pass_filter(1200),
        ),
        (
            "high_pass_filter, 10 s stereo",
            lambda: ten.high_pass_filter(1200),
            lambda: py_ten.high_pass_filter(1200),
        ),
        (
            "speedup 1.5x, 60 s stereo",
            lambda: minute.speedup(1.5),
            lambda: py_minute.speedup(1.5),
        ),
    ]


def main() -> None:
    print(f"Machine: {cpu_name()} ({platform.machine()}, {platform.system()})")
    print()
    print("| operation | mojo-pydub | pydub | speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_seconds = timeit(ours)
        python_seconds = timeit(upstream)
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{python_seconds * 1000:.2f} ms | "
            f"{python_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
