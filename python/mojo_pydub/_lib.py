"""ctypes bindings for the Mojo PCM kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYDUB_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pydub.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpd_gain": ([I, I, I, I, F], None),
    "mpd_mix": ([I, I, I, I, I, F], None),
    "mpd_peak": ([I, I, I], I),
    "mpd_rms": ([I, I, I], I),
    "mpd_reverse": ([I, I, I, I], None),
    "mpd_fade": ([I, I, I, I, I, F, F, I, I, I, I, I], None),
    "mpd_low_pass": ([I, I, I, I, I, F], None),
    "mpd_high_pass": ([I, I, I, I, I, F], None),
    "mpd_gain_stereo": ([I, I, I, I, I, F, F], None),
}

_DTYPES = {
    1: np.dtype("i1"),
    2: np.dtype("<i2"),
    4: np.dtype("<i4"),
}

_library: ctypes.CDLL | None = None

_bytes_address = ctypes.pythonapi.PyBytes_AsString
_bytes_address.argtypes = [ctypes.py_object]
_bytes_address.restype = ctypes.c_void_p

_new_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_new_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes.restype = ctypes.py_object


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def samples(data: bytes, sample_width: int) -> np.ndarray:
    try:
        dtype = _DTYPES[sample_width]
    except KeyError:
        raise ValueError("sample_width must be 1, 2, or 4 bytes") from None
    if len(data) % sample_width:
        raise ValueError("PCM byte length must be a multiple of sample_width")
    return np.frombuffer(data, dtype=dtype)


def empty(count: int, sample_width: int) -> np.ndarray:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    try:
        dtype = _DTYPES[sample_width]
    except KeyError:
        raise ValueError("sample_width must be 1, 2, or 4 bytes") from None
    return np.empty(count, dtype=dtype)


def address(array: np.ndarray) -> int:
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if array.size == 0:
        raise ValueError("cannot take a native address for an empty array")
    return int(array.ctypes.data)


def bytes_address(data: bytes) -> int:
    if not isinstance(data, bytes):
        raise TypeError("native PCM inputs must be bytes")
    if not data:
        raise ValueError("cannot take a native address for an empty buffer")
    pointer = _bytes_address(data)
    if not pointer:
        raise RuntimeError("CPython did not provide a buffer address")
    return int(pointer)


def output_buffer(size: int) -> tuple[bytes, int]:
    if size <= 0:
        raise ValueError("native output size must be positive")
    data = _new_bytes(None, size)
    return data, bytes_address(data)


def gain(data: bytes, width: int, factor: float) -> bytes:
    source = samples(data, width)
    if not data:
        return b""
    destination, destination_address = output_buffer(len(data))
    lib().mpd_gain(
        bytes_address(data),
        destination_address,
        source.size,
        width,
        factor,
    )
    return destination


def mix(a: bytes, b: bytes, width: int, a_factor: float = 1.0) -> bytes:
    a_samples = samples(a, width)
    b_samples = samples(b, width)
    if len(a) != len(b):
        raise ValueError("mix buffers must have equal sample counts")
    if not a:
        return b""
    destination, destination_address = output_buffer(len(a))
    lib().mpd_mix(
        bytes_address(a),
        bytes_address(b),
        destination_address,
        a_samples.size,
        width,
        a_factor,
    )
    assert a_samples.size == b_samples.size
    return destination


def overlay(
    base: bytes,
    over: bytes,
    position: int,
    times: int,
    width: int,
    base_factor: float,
) -> bytes:
    samples(base, width)
    samples(over, width)
    if position < 0 or position % width:
        raise ValueError("overlay position must be a non-negative sample boundary")
    if not base or not over or not times or position >= len(base):
        return base
    destination, destination_address = output_buffer(len(base))
    base_address = bytes_address(base)
    over_address = bytes_address(over)
    ctypes.memmove(destination_address, base_address, len(base))
    while times and position < len(base):
        size = min(len(over), len(base) - position)
        lib().mpd_mix(
            base_address + position,
            over_address,
            destination_address + position,
            size // width,
            width,
            base_factor,
        )
        position += size
        times -= 1
    return destination


def peak(data: bytes, width: int) -> int:
    source = samples(data, width)
    if source.size == 0:
        return 0
    return int(lib().mpd_peak(address(source), source.size, width))


def rms(data: bytes, width: int) -> int:
    source = samples(data, width)
    if source.size == 0:
        return 0
    return int(lib().mpd_rms(address(source), source.size, width))


def reverse(data: bytes, width: int) -> bytes:
    source = samples(data, width)
    if source.size == 0:
        return b""
    destination = empty(source.size, width)
    lib().mpd_reverse(address(source), address(destination), source.size, width)
    return destination.tobytes()


def fade(
    data: bytes,
    width: int,
    frames: int,
    channels: int,
    from_factor: float,
    to_factor: float,
    frame_rate: int,
    start_frame: int,
    start_ms: int,
    duration_ms: int,
    coarse: bool,
) -> bytes:
    source = samples(data, width)
    if frames < 0 or channels < 1 or frames * channels != source.size:
        raise ValueError("fade dimensions do not match the PCM buffer")
    if frame_rate < 1 or start_frame < 0 or duration_ms <= 0:
        raise ValueError("invalid fade timing")
    if source.size == 0:
        return b""
    destination = empty(source.size, width)
    lib().mpd_fade(
        address(source),
        address(destination),
        frames,
        channels,
        width,
        from_factor,
        to_factor,
        frame_rate,
        start_frame,
        start_ms,
        duration_ms,
        int(coarse),
    )
    return destination.tobytes()


def filter_pcm(
    name: str,
    data: bytes,
    width: int,
    frames: int,
    channels: int,
    alpha: float,
) -> bytes:
    source = samples(data, width)
    if frames < 0 or channels < 1 or frames * channels != source.size:
        raise ValueError("filter dimensions do not match the PCM buffer")
    if source.size == 0:
        return b""
    destination = empty(source.size, width)
    getattr(lib(), name)(
        address(source), address(destination), frames, channels, width, alpha
    )
    return destination.tobytes()


def gain_stereo(
    data: bytes,
    width: int,
    frames: int,
    channels: int,
    left_factor: float,
    right_factor: float,
) -> bytes:
    source = samples(data, width)
    if channels not in (1, 2) or frames < 0 or frames * channels != source.size:
        raise ValueError("stereo gain dimensions do not match the PCM buffer")
    if source.size == 0:
        return b""
    destination = empty(frames * 2, width)
    lib().mpd_gain_stereo(
        address(source),
        address(destination),
        frames,
        channels,
        width,
        left_factor,
        right_factor,
    )
    return destination.tobytes()
