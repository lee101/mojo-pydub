from __future__ import annotations

import array
import io
import math
import wave
from pathlib import Path

import numpy as np

from . import _lib
from .utils import db_to_float, ratio_to_db

_ARRAY_TYPES = {1: "b", 2: "h", 4: "i"}


class AudioSegment:
    """Immutable, pydub-compatible signed integer PCM audio."""

    def __init__(
        self,
        data: bytes | bytearray | memoryview | array.array | None = None,
        *args,
        sample_width: int | None = None,
        frame_rate: int | None = None,
        channels: int | None = None,
        **kwargs,
    ):
        metadata = kwargs.pop("metadata", None)
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument {name!r}")
        if metadata is not None:
            sample_width = metadata["sample_width"]
            frame_rate = metadata["frame_rate"]
            channels = metadata["channels"]
        if data is None:
            data = b""
        if isinstance(data, array.array):
            data = data.tobytes()
        elif hasattr(data, "read"):
            data = data.read()
        data = bytes(data)
        if sample_width is None or frame_rate is None or channels is None:
            raise ValueError(
                "sample_width, frame_rate, and channels must all be specified"
            )
        if sample_width not in _ARRAY_TYPES:
            raise ValueError("sample_width must be 1, 2, or 4 bytes")
        if channels < 1 or frame_rate < 1:
            raise ValueError("channels and frame_rate must be positive")
        frame_width = sample_width * channels
        if len(data) % frame_width:
            raise ValueError(
                "data length must be a multiple of sample_width * channels"
            )
        self.sample_width = int(sample_width)
        self.frame_rate = int(frame_rate)
        self.channels = int(channels)
        self.frame_width = frame_width
        self._data = data

    @property
    def raw_data(self) -> bytes:
        return self._data

    @property
    def array_type(self) -> str:
        return _ARRAY_TYPES[self.sample_width]

    def get_array_of_samples(self, array_type_override=None) -> array.array:
        return array.array(array_type_override or self.array_type, self._data)

    def __len__(self) -> int:
        return round(1000 * self.frame_count() / self.frame_rate)

    def __eq__(self, other) -> bool:
        return isinstance(other, AudioSegment) and self._data == other._data

    def __hash__(self) -> int:
        return hash(
            (
                AudioSegment,
                self.channels,
                self.frame_rate,
                self.sample_width,
                self._data,
            )
        )

    def _spawn(self, data, overrides=None):
        metadata = {
            "sample_width": self.sample_width,
            "frame_rate": self.frame_rate,
            "channels": self.channels,
        }
        if overrides:
            metadata.update(overrides)
        return self.__class__(data=data, **metadata)

    def frame_count(self, ms=None) -> float:
        if ms is not None:
            return ms * (self.frame_rate / 1000.0)
        return float(len(self._data) // self.frame_width)

    def _parse_position(self, value) -> int:
        if value < 0:
            value = len(self) - abs(value)
        if value == float("inf"):
            value = len(self)
        return int(self.frame_count(ms=value))

    def __getitem__(self, millisecond):
        if isinstance(millisecond, slice):
            if millisecond.step:
                return (
                    self[i : i + millisecond.step]
                    for i in range(*millisecond.indices(len(self)))
                )
            start = 0 if millisecond.start is None else millisecond.start
            end = len(self) if millisecond.stop is None else millisecond.stop
            start = min(start, len(self))
            end = min(end, len(self))
        else:
            start, end = millisecond, millisecond + 1
        start_byte = max(0, self._parse_position(start)) * self.frame_width
        end_byte = max(0, self._parse_position(end)) * self.frame_width
        data = self._data[start_byte:end_byte]
        expected_length = end_byte - start_byte
        missing_frames = (expected_length - len(data)) // self.frame_width
        if missing_frames:
            if missing_frames > self.frame_count(ms=2):
                raise ValueError("slice would require more than 2 ms of padding")
            data += bytes(self.frame_width) * missing_frames
        return self._spawn(data)

    def get_frame(self, index: int) -> bytes:
        start = index * self.frame_width
        return self._data[start : start + self.frame_width]

    def get_sample_slice(self, start_sample=None, end_sample=None):
        frames = int(self.frame_count())
        start = max(0, min(frames, 0 if start_sample is None else start_sample))
        end = max(0, min(frames, frames if end_sample is None else end_sample))
        return self._spawn(
            self._data[start * self.frame_width : end * self.frame_width]
        )

    @classmethod
    def empty(cls):
        return cls(b"", sample_width=1, frame_rate=1, channels=1)

    @classmethod
    def silent(cls, duration=1000, frame_rate=11025):
        frames = int(frame_rate * duration / 1000.0)
        return cls(
            b"\0\0" * frames,
            sample_width=2,
            frame_rate=frame_rate,
            channels=1,
        )

    @classmethod
    def from_mono_audiosegments(cls, *mono_segments):
        if not mono_segments:
            raise ValueError("At least one AudioSegment instance is required")
        segments = cls._sync(*mono_segments)
        if segments[0].channels != 1:
            raise ValueError(
                "AudioSegment.from_mono_audiosegments requires mono segments"
            )
        frames = max(int(segment.frame_count()) for segment in segments)
        dtype = _lib.samples(b"", segments[0].sample_width).dtype
        joined = np.zeros((frames, len(segments)), dtype=dtype)
        for channel, segment in enumerate(segments):
            values = _lib.samples(segment.raw_data, segment.sample_width)
            joined[: values.size, channel] = values
        return cls(
            joined.tobytes(),
            sample_width=segments[0].sample_width,
            frame_rate=segments[0].frame_rate,
            channels=len(segments),
        )

    @classmethod
    def from_wav(cls, file):
        return cls.from_file(file, format="wav")

    @classmethod
    def from_file(cls, file, format=None, **kwargs):
        if format not in (None, "wav", "wave"):
            raise NotImplementedError("only uncompressed PCM WAV input is covered")
        with wave.open(str(file) if isinstance(file, (str, Path)) else file, "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError("compressed WAV is not supported")
            sample_width = wav.getsampwidth()
            data = wav.readframes(wav.getnframes())
            if sample_width == 1:
                data = bytes((value - 128) & 0xFF for value in data)
            return cls(
                data,
                sample_width=sample_width,
                frame_rate=wav.getframerate(),
                channels=wav.getnchannels(),
            )

    def export(self, out_f=None, format="wav", **kwargs):
        if format not in ("wav", "wave"):
            raise NotImplementedError("only uncompressed PCM WAV output is covered")
        target = io.BytesIO() if out_f is None else out_f
        close_after = False
        if isinstance(target, (str, Path)):
            target = open(target, "w+b")
            close_after = True
        with wave.open(target, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.sample_width)
            wav.setframerate(self.frame_rate)
            data = self._data
            if self.sample_width == 1:
                data = bytes((value + 128) & 0xFF for value in data)
            wav.writeframes(data)
        if close_after:
            target.close()
            return out_f
        target.seek(0)
        return target

    @classmethod
    def _sync(cls, *segments):
        channels = max(segment.channels for segment in segments)
        frame_rate = max(segment.frame_rate for segment in segments)
        sample_width = max(segment.sample_width for segment in segments)
        return tuple(
            segment.set_channels(channels)
            .set_frame_rate(frame_rate)
            .set_sample_width(sample_width)
            for segment in segments
        )

    def set_sample_width(self, sample_width):
        if sample_width == self.sample_width:
            return self
        if sample_width not in _ARRAY_TYPES:
            raise ValueError("sample_width must be 1, 2, or 4 bytes")
        values = _lib.samples(self._data, self.sample_width).astype(np.int64)
        shift = 8 * (sample_width - self.sample_width)
        if shift > 0:
            values <<= shift
        else:
            values >>= -shift
        limits = (-(1 << (sample_width * 8 - 1)), (1 << (sample_width * 8 - 1)) - 1)
        dtype = {1: np.int8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}[sample_width]
        converted = np.clip(values, *limits).astype(dtype)
        return self._spawn(converted.tobytes(), {"sample_width": sample_width})

    def set_frame_rate(self, frame_rate):
        frame_rate = int(frame_rate)
        if frame_rate == self.frame_rate:
            return self
        frames = int(self.frame_count())
        if frames == 0:
            return self._spawn(b"", {"frame_rate": frame_rate})
        source = _lib.samples(self._data, self.sample_width).reshape(
            frames, self.channels
        )
        destination_frames = max(1, int(frames * frame_rate / self.frame_rate))
        positions = np.arange(destination_frames) * self.frame_rate / frame_rate
        indices = np.minimum(positions.astype(np.int64), frames - 1)
        converted = source[indices]
        return self._spawn(converted.tobytes(), {"frame_rate": frame_rate})

    def set_channels(self, channels):
        channels = int(channels)
        if channels == self.channels:
            return self
        frames = int(self.frame_count())
        source = _lib.samples(self._data, self.sample_width).reshape(
            frames, self.channels
        )
        if self.channels == 1:
            converted = np.repeat(source, channels, axis=1)
        elif channels == 1:
            converted = np.floor(
                source.astype(np.float64).sum(axis=1, keepdims=True)
                / self.channels
            ).astype(source.dtype)
        else:
            raise ValueError(
                "set_channels supports mono-to-multi and multi-to-mono conversion"
            )
        return self._spawn(converted.tobytes(), {"channels": channels})

    def split_to_mono(self):
        if self.channels == 1:
            return [self]
        frames = int(self.frame_count())
        values = _lib.samples(self._data, self.sample_width).reshape(
            frames, self.channels
        )
        return [
            self._spawn(
                np.ascontiguousarray(values[:, channel]).tobytes(),
                {"channels": 1},
            )
            for channel in range(self.channels)
        ]

    @property
    def rms(self) -> int:
        return _lib.rms(self._data, self.sample_width)

    @property
    def dBFS(self) -> float:
        return (
            ratio_to_db(self.rms / self.max_possible_amplitude)
            if self.rms
            else -float("inf")
        )

    @property
    def max(self) -> int:
        return _lib.peak(self._data, self.sample_width)

    @property
    def max_possible_amplitude(self) -> float:
        return float(1 << (self.sample_width * 8 - 1))

    @property
    def max_dBFS(self) -> float:
        return ratio_to_db(self.max, self.max_possible_amplitude)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count() / self.frame_rate if self.frame_rate else 0.0

    def apply_gain(self, volume_change):
        return self._spawn(
            _lib.gain(
                self._data,
                self.sample_width,
                db_to_float(float(volume_change)),
            )
        )

    def overlay(
        self,
        seg,
        position=0,
        loop=False,
        times=None,
        gain_during_overlay=None,
    ):
        if loop:
            times = -1
        elif times is None:
            times = 1
        elif times == 0:
            return self._spawn(self._data)
        base, over = self._sync(self, seg)
        # pydub's millisecond slice used by overlay pads a sub-millisecond
        # partial final frame interval with silence.
        base = base[:]
        position = min(position, len(base))
        position_bytes = (
            max(0, base._parse_position(position)) * base.frame_width
        )
        factor = (
            db_to_float(float(gain_during_overlay))
            if gain_during_overlay
            else 1.0
        )
        return base._spawn(
            _lib.overlay(
                base._data,
                over._data,
                position_bytes,
                times,
                base.sample_width,
                factor,
            )
        )

    def append(self, seg, crossfade=100):
        first, second = self._sync(self, seg)
        if not crossfade:
            return first._spawn(first._data + second._data)
        if crossfade > len(first) or crossfade > len(second):
            raise ValueError("Crossfade is longer than an AudioSegment")
        left = first[-crossfade:].fade(
            to_gain=-120, start=0, end=float("inf")
        )
        right = second[:crossfade].fade(
            from_gain=-120, start=0, end=float("inf")
        )
        mixed = left.overlay(right)
        return first._spawn(
            first[:-crossfade]._data + mixed._data + second[crossfade:]._data
        )

    def fade(
        self,
        to_gain=0,
        from_gain=0,
        start=None,
        end=None,
        duration=None,
    ):
        if None not in (duration, end, start):
            raise TypeError('Only two of "start", "end", and "duration" may be specified')
        if to_gain == 0 and from_gain == 0:
            return self
        start = min(len(self), start) if start is not None else None
        end = min(len(self), end) if end is not None else None
        if start is not None and start < 0:
            start += len(self)
        if end is not None and end < 0:
            end += len(self)
        if duration is not None and duration < 0:
            raise ValueError("duration must be a positive integer")
        if duration:
            if start is not None:
                end = start + duration
            elif end is not None:
                start = end - duration
        else:
            duration = end - start
        start_frame = self._parse_position(start)
        end_frame = self._parse_position(end)
        before = self._data[: start_frame * self.frame_width]
        region = self._data[
            start_frame * self.frame_width : end_frame * self.frame_width
        ]
        after = self._data[end_frame * self.frame_width :]
        if from_gain:
            before = _lib.gain(
                before, self.sample_width, db_to_float(from_gain)
            )
        frames = end_frame - start_frame
        if frames:
            region = _lib.fade(
                region,
                self.sample_width,
                frames,
                self.channels,
                db_to_float(from_gain),
                db_to_float(to_gain),
                self.frame_rate,
                start_frame,
                int(start),
                int(duration),
                duration > 100,
            )
        if to_gain:
            after = _lib.gain(after, self.sample_width, db_to_float(to_gain))
        return self._spawn(before + region + after)

    def fade_in(self, duration):
        return self.fade(from_gain=-120, start=0, duration=duration)

    def fade_out(self, duration):
        return self.fade(to_gain=-120, end=float("inf"), duration=duration)

    def reverse(self):
        return self._spawn(_lib.reverse(self._data, self.sample_width))

    def __add__(self, value):
        return self.append(value, crossfade=0) if isinstance(
            value, AudioSegment
        ) else self.apply_gain(value)

    def __radd__(self, value):
        if value == 0:
            return self
        raise TypeError("Gains must be the second addend after AudioSegment")

    def __sub__(self, value):
        if isinstance(value, AudioSegment):
            raise TypeError("AudioSegment objects cannot be subtracted")
        return self.apply_gain(-value)

    def __mul__(self, value):
        if isinstance(value, AudioSegment):
            return self.overlay(value, loop=True)
        return self._spawn(self._data * int(value))

    __rmul__ = __mul__
