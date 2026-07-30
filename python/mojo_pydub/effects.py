from __future__ import annotations

import math

from . import _lib
from .audio_segment import AudioSegment
from .utils import db_to_float, ratio_to_db


def normalize(seg: AudioSegment, headroom=0.1):
    peak = seg.max
    if peak == 0:
        return seg
    target = seg.max_possible_amplitude * db_to_float(-headroom)
    return seg.apply_gain(ratio_to_db(target / peak))


def invert_phase(seg: AudioSegment, channels=(1, 1)):
    if channels == (1, 1):
        return seg._spawn(_lib.gain(seg.raw_data, seg.sample_width, -1.0))
    if seg.channels != 2:
        raise Exception(
            "Can't implicitly convert an AudioSegment with "
            f"{seg.channels} channels to stereo."
        )
    left, right = seg.split_to_mono()
    if channels == (1, 0):
        left = invert_phase(left)
    else:
        right = invert_phase(right)
    return AudioSegment.from_mono_audiosegments(left, right)


def low_pass_filter(seg: AudioSegment, cutoff):
    rc = 1.0 / (float(cutoff) * 2 * math.pi)
    dt = 1.0 / seg.frame_rate
    alpha = dt / (rc + dt)
    data = _lib.filter_pcm(
        "mpd_low_pass",
        seg.raw_data,
        seg.sample_width,
        int(seg.frame_count()),
        seg.channels,
        alpha,
    )
    return seg._spawn(data)


def high_pass_filter(seg: AudioSegment, cutoff):
    rc = 1.0 / (float(cutoff) * 2 * math.pi)
    dt = 1.0 / seg.frame_rate
    alpha = rc / (rc + dt)
    data = _lib.filter_pcm(
        "mpd_high_pass",
        seg.raw_data,
        seg.sample_width,
        int(seg.frame_count()),
        seg.channels,
        alpha,
    )
    return seg._spawn(data)


def apply_gain_stereo(seg: AudioSegment, left_gain=0.0, right_gain=0.0):
    if seg.channels not in (1, 2):
        raise ValueError("apply_gain_stereo supports mono or stereo audio")
    data = _lib.gain_stereo(
        seg.raw_data,
        seg.sample_width,
        int(seg.frame_count()),
        seg.channels,
        db_to_float(left_gain),
        db_to_float(right_gain),
    )
    return seg._spawn(data, {"channels": 2})


def pan(seg: AudioSegment, pan_amount):
    if not -1.0 <= pan_amount <= 1.0:
        raise ValueError("pan_amount should be between -1.0 and +1.0")
    max_boost_db = ratio_to_db(2.0)
    boost_db = abs(pan_amount) * max_boost_db
    boost_factor = db_to_float(boost_db)
    reduce_factor = db_to_float(max_boost_db) - boost_factor
    reduce_db = ratio_to_db(reduce_factor)
    boost_db /= 2.0
    if pan_amount < 0:
        return apply_gain_stereo(seg, boost_db, reduce_db)
    return apply_gain_stereo(seg, reduce_db, boost_db)


def speedup(seg: AudioSegment, playback_speed=1.5, chunk_size=150, crossfade=25):
    keep = 1.0 / playback_speed
    if playback_speed < 2.0:
        remove = int(chunk_size * (1 - keep) / keep)
    else:
        remove = int(chunk_size)
        chunk_size = int(keep * chunk_size / (1 - keep))
    crossfade = min(crossfade, remove - 1)
    full_chunk = chunk_size + remove
    chunks = [seg[start : start + full_chunk] for start in range(0, len(seg), full_chunk)]
    if len(chunks) < 2:
        raise Exception("Could not speed up AudioSegment: it was too short")
    remove -= crossfade
    last = chunks.pop()
    chunks = [chunk[:-remove] for chunk in chunks]
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result.append(chunk, crossfade=crossfade)
    return result + last


for _effect in (
    normalize,
    invert_phase,
    low_pass_filter,
    high_pass_filter,
    apply_gain_stereo,
    pan,
    speedup,
):
    setattr(AudioSegment, _effect.__name__, _effect)
