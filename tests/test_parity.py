import io

import numpy as np
import pytest
from pydub import AudioSegment as PyAudioSegment

from mojo_pydub import AudioSegment
from mojo_pydub import effects
from mojo_pydub import _lib

RNG = np.random.default_rng(2026)
DTYPES = {1: np.int8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}


def pair(values, width=2, frame_rate=8000, channels=1):
    data = np.asarray(values, dtype=DTYPES[width]).tobytes()
    kwargs = {
        "sample_width": width,
        "frame_rate": frame_rate,
        "channels": channels,
    }
    return AudioSegment(data, **kwargs), PyAudioSegment(data, **kwargs)


def random_pair(frames=8000, width=2, frame_rate=8000, channels=1):
    bits = width * 8
    values = RNG.integers(
        -(1 << (bits - 1)),
        1 << (bits - 1),
        size=frames * channels,
        dtype=DTYPES[width],
    )
    return pair(values, width, frame_rate, channels)


def assert_parity(ours, upstream):
    assert ours.raw_data == upstream.raw_data
    assert ours.sample_width == upstream.sample_width
    assert ours.frame_rate == upstream.frame_rate
    assert ours.channels == upstream.channels


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("gain", [-12.0, -3.5, 0.0, 2.0])
def test_apply_gain_matches_pydub(width, gain):
    ours, upstream = random_pair(4099, width=width)
    assert_parity(ours.apply_gain(gain), upstream.apply_gain(gain))


@pytest.mark.parametrize("width", [1, 2, 4])
def test_peak_rms_and_levels_match_pydub(width):
    ours, upstream = random_pair(10007, width=width, channels=2)
    assert ours.max == upstream.max
    assert ours.rms == upstream.rms
    assert ours.dBFS == pytest.approx(upstream.dBFS)
    assert ours.max_dBFS == pytest.approx(upstream.max_dBFS)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_normalize_matches_pydub(width):
    ours, upstream = random_pair(10001, width=width)
    assert_parity(ours.normalize(headroom=0.25), upstream.normalize(headroom=0.25))


def test_normalize_silence_is_noop():
    ours = AudioSegment.silent(250, frame_rate=8000)
    assert ours.normalize() is ours


@pytest.mark.parametrize("width", [1, 2, 4])
def test_overlay_once_loop_times_and_ducking_match(width):
    base, base_ref = random_pair(8000, width=width, channels=2)
    over, over_ref = random_pair(1200, width=width, channels=2)
    cases = [
        ({}, {}),
        ({"position": 125}, {"position": 125}),
        (
            {"position": 125, "times": 3, "gain_during_overlay": -7.0},
            {"position": 125, "times": 3, "gain_during_overlay": -7.0},
        ),
        (
            {"position": 125, "loop": True},
            {"position": 125, "loop": True},
        ),
    ]
    for ours_kwargs, ref_kwargs in cases:
        assert_parity(
            base.overlay(over, **ours_kwargs),
            base_ref.overlay(over_ref, **ref_kwargs),
        )


def test_overlay_zero_times_is_immutable_noop():
    base, _ = random_pair(100)
    result = base.overlay(base, times=0)
    assert result is not base
    assert result.raw_data == base.raw_data


@pytest.mark.parametrize("width", [1, 2, 4])
def test_mix_simd_remainder_matches_pydub(width):
    ours, upstream = random_pair(4103, width=width)
    over, over_ref = random_pair(4103, width=width)
    assert_parity(ours.overlay(over), upstream.overlay(over_ref))


@pytest.mark.parametrize("samples", [1_048_575, 1_048_579])
def test_gain_and_overlay_match_across_parallel_threshold_with_simd_tail(samples):
    ours, upstream = random_pair(samples, width=2, frame_rate=1000)
    over, over_ref = random_pair(samples, width=2, frame_rate=1000)
    assert_parity(ours.apply_gain(-3.5), upstream.apply_gain(-3.5))
    assert_parity(ours.overlay(over), upstream.overlay(over_ref))


@pytest.mark.parametrize("duration", [25, 200, 500])
def test_fade_in_and_out_match_short_and_coarse_pydub_paths(duration):
    ours, upstream = random_pair(8000, frame_rate=8000, channels=2)
    assert_parity(ours.fade_in(duration), upstream.fade_in(duration))
    assert_parity(ours.fade_out(duration), upstream.fade_out(duration))


def test_partial_fade_matches_pydub():
    ours, upstream = random_pair(12000, frame_rate=12000)
    assert_parity(
        ours.fade(from_gain=-9, to_gain=2, start=100, duration=350),
        upstream.fade(from_gain=-9, to_gain=2, start=100, duration=350),
    )


@pytest.mark.parametrize("crossfade", [0, 25, 200])
def test_append_matches_pydub(crossfade):
    first, first_ref = random_pair(5000, frame_rate=8000)
    second, second_ref = random_pair(4000, frame_rate=8000)
    assert_parity(
        first.append(second, crossfade=crossfade),
        first_ref.append(second_ref, crossfade=crossfade),
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_low_pass_filter_matches_pydub(width, channels):
    ours, upstream = random_pair(
        3000, width=width, frame_rate=44100, channels=channels
    )
    assert_parity(
        effects.low_pass_filter(ours, 1200),
        upstream.low_pass_filter(1200),
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_high_pass_filter_matches_pydub(width, channels):
    ours, upstream = random_pair(
        3000, width=width, frame_rate=44100, channels=channels
    )
    assert_parity(
        effects.high_pass_filter(ours, 1200),
        upstream.high_pass_filter(1200),
    )


@pytest.mark.parametrize("channels", [(1, 1), (1, 0), (0, 1)])
def test_invert_phase_matches_pydub(channels):
    ours, upstream = random_pair(3000, channels=2)
    assert_parity(
        effects.invert_phase(ours, channels=channels),
        upstream.invert_phase(channels=channels),
    )


@pytest.mark.parametrize("source_channels", [1, 2])
@pytest.mark.parametrize("left_gain,right_gain", [(-3, 2), (0, 0), (6, -12)])
def test_apply_gain_stereo_matches_pydub(
    source_channels, left_gain, right_gain
):
    ours, upstream = random_pair(3000, channels=source_channels)
    assert_parity(
        ours.apply_gain_stereo(left_gain, right_gain),
        upstream.apply_gain_stereo(left_gain, right_gain),
    )


@pytest.mark.parametrize("amount", [-1.0, -0.35, 0.0, 0.4, 1.0])
def test_pan_matches_pydub(amount):
    ours, upstream = random_pair(3000, channels=1)
    assert_parity(ours.pan(amount), upstream.pan(amount))


def test_pan_rejects_out_of_range_amount():
    segment = AudioSegment.silent(10)
    with pytest.raises(ValueError):
        segment.pan(1.01)


@pytest.mark.parametrize("speed", [1.25, 1.5, 2.5])
def test_speedup_matches_pydub(speed):
    ours, upstream = random_pair(24000, frame_rate=8000)
    assert_parity(ours.speedup(speed), upstream.speedup(speed))


def test_reverse_and_slice_match_pydub():
    ours, upstream = random_pair(4000, frame_rate=8000, channels=2)
    assert_parity(ours.reverse(), upstream.reverse())
    assert_parity(ours[125:375], upstream[125:375])
    assert_parity(ours[-200:], upstream[-200:])


def test_sample_slice_concatenation_and_duration_match_pydub():
    ours, upstream = random_pair(4000, frame_rate=8000, channels=2)
    assert_parity(
        ours.get_sample_slice(123, 987),
        upstream.get_sample_slice(123, 987),
    )
    assert_parity(ours + ours, upstream + upstream)
    assert ours.duration_seconds == upstream.duration_seconds


def test_split_and_join_mono_matches_pydub():
    ours, upstream = random_pair(1000, channels=2)
    ours_channels = ours.split_to_mono()
    upstream_channels = upstream.split_to_mono()
    for got, expected in zip(ours_channels, upstream_channels):
        assert_parity(got, expected)
    assert_parity(
        AudioSegment.from_mono_audiosegments(*ours_channels),
        PyAudioSegment.from_mono_audiosegments(*upstream_channels),
    )


@pytest.mark.parametrize("target_width", [1, 2, 4])
def test_sample_width_conversion_matches_pydub(target_width):
    ours, upstream = random_pair(1000, width=2)
    assert_parity(
        ours.set_sample_width(target_width),
        upstream.set_sample_width(target_width),
    )


@pytest.mark.parametrize("source_channels,target_channels", [(1, 2), (1, 4), (2, 1)])
def test_channel_conversion_matches_pydub(source_channels, target_channels):
    ours, upstream = random_pair(1000, channels=source_channels)
    assert_parity(
        ours.set_channels(target_channels),
        upstream.set_channels(target_channels),
    )


def test_nearest_neighbor_frame_rate_conversion_is_well_formed():
    ours, _ = random_pair(1000, frame_rate=8000, channels=2)
    converted = ours.set_frame_rate(12000)
    assert converted.frame_rate == 12000
    assert converted.channels == ours.channels
    assert converted.sample_width == ours.sample_width
    assert len(converted.raw_data) % converted.frame_width == 0


def test_wav_roundtrip_matches_raw_pcm():
    ours, _ = random_pair(1000, channels=2)
    encoded = ours.export(format="wav")
    decoded = AudioSegment.from_wav(encoded)
    assert_parity(decoded, ours)


def test_unsigned_8bit_wav_roundtrip():
    ours, _ = random_pair(1000, width=1)
    encoded = ours.export(format="wav")
    assert AudioSegment.from_wav(io.BytesIO(encoded.read())).raw_data == ours.raw_data


def test_silent_empty_and_repeat_behaviour():
    silent = AudioSegment.silent(duration=125, frame_rate=8000)
    assert len(silent) == 125
    assert silent.max == silent.rms == 0
    assert AudioSegment.empty().raw_data == b""
    assert (silent * 3).raw_data == silent.raw_data * 3


def test_constructor_validates_pcm_shape_and_width():
    with pytest.raises(ValueError):
        AudioSegment(b"\0", sample_width=2, frame_rate=8000, channels=1)
    with pytest.raises(ValueError):
        AudioSegment(b"", sample_width=3, frame_rate=8000, channels=1)


def test_ffi_wrappers_reject_misaligned_or_inconsistent_buffers():
    with pytest.raises(ValueError, match="multiple"):
        _lib.gain(b"\0", 2, 1.0)
    with pytest.raises(ValueError, match="dimensions"):
        _lib.filter_pcm("mpd_low_pass", b"\0\0", 2, 2, 1, 0.5)
    with pytest.raises(ValueError, match="sample boundary"):
        _lib.overlay(b"\0\0", b"\0\0", 1, 1, 2, 1.0)
