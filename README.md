# mojo-pydub

`mojo-pydub` is a standalone Mojo implementation of the compute-heavy PCM
mixing and effects in Python's [pydub](https://github.com/jiaaro/pydub). It
keeps pydub's immutable `AudioSegment` model and the names and signatures of
the covered methods, while moving sample loops into one compiled shared
library.

This is a focused port, not a replacement for every media feature in pydub.
The Python import is `mojo_pydub`, so it can be installed beside `pydub` for
parity testing.

## Covered subset

The following operations match upstream pydub sample-for-sample on signed
8-, 16-, and 32-bit integer PCM:

- `AudioSegment`: construction from raw PCM, millisecond slicing,
  `silent`, `empty`, `from_mono_audiosegments`, `split_to_mono`,
  `get_sample_slice`, `set_sample_width`, mono/multichannel conversion,
  repetition, concatenation, `reverse`, `rms`, `max`, `dBFS`, and `max_dBFS`
- Mixing: `apply_gain`, `overlay` (including `loop`, `times`, and ducking),
  `append`, short sample-accurate fades, and pydub's coarse fades over 100 ms
- Effects: `normalize`, `invert_phase`, `low_pass_filter`,
  `high_pass_filter`, `apply_gain_stereo`, `pan`, and `speedup`
- Uncompressed PCM WAV input and output, including WAV's unsigned 8-bit
  representation

The test suite compares raw output bytes against the real upstream `pydub`
package. It covers mono, stereo, and four-channel buffers where the upstream
effect supports them.

Not covered:

- MP3, AAC, FLAC, Ogg, or other ffmpeg-backed decoding and encoding
- silence detection/splitting, dynamic-range compression, DC-offset
  correction, and user-defined channel filters
- pydub's full export metadata, tags, codecs, and subprocess integration
- exact `audioop.ratecv` resampling; `set_frame_rate` is available for format
  synchronization but currently uses nearest-neighbor resampling
- floating-point PCM and 24-bit packed PCM

## Install and run

The project pins the Mojo nightly dialect it was developed against.

```bash
pixi install
pixi run build
pixi run test
```

`pydub` is an environment dependency only so the parity tests and benchmarks
can run. The `mojo_pydub` package itself uses NumPy and the built Mojo shared
library, not pydub.

## Usage

This example creates a one-second tone, mixes in a quieter copy, filters it,
and writes an ordinary WAV file:

```python
import numpy as np
from mojo_pydub import AudioSegment

rate = 44_100
t = np.arange(rate) / rate
pcm = (12_000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")

tone = AudioSegment(
    pcm.tobytes(),
    sample_width=2,
    frame_rate=rate,
    channels=1,
)
mixed = (
    tone.fade_in(100)
    .overlay(tone.apply_gain(-6), position=250)
    .low_pass_filter(3_000)
)
mixed.export("mixed.wav", format="wav")
```

## Benchmarks

Measured with `pixi run bench`, which takes a machine-wide flock before
running. Times are the best of five warm runs on an Intel Xeon E5-2697 v4 at
2.30 GHz (`x86_64`, Linux). A speedup below `1.00x` means upstream pydub was
faster.

| operation | mojo-pydub | pydub | speedup |
|---|---:|---:|---:|
| apply_gain, 60 s stereo | 22.03 ms | 22.42 ms | 1.02x |
| normalize, 60 s stereo | 25.50 ms | 44.88 ms | 1.76x |
| overlay, 60 s + 30 s stereo | 24.74 ms | 31.78 ms | 1.28x |
| fade_in, 10 s stereo | 3.12 ms | 51.35 ms | 16.45x |
| pan, 60 s stereo | 17.63 ms | 199.55 ms | 11.32x |
| low_pass_filter, 10 s stereo | 3.63 ms | 592.59 ms | 163.19x |
| high_pass_filter, 10 s stereo | 3.33 ms | 826.04 ms | 248.32x |
| speedup 1.5x, 60 s stereo | 362.64 ms | 1427.96 ms | 3.94x |

Gain and mixing use float64-width SIMD with scalar remainder handling. Buffers
of at least 1,048,576 samples are split into independent 262,144-sample chunks
for CPU parallelism; smaller inputs remain serial. The Python binding allocates
a writable output buffer, keeps it alive for the native call, and converts it to
immutable `bytes` afterward. Overlay copies the base once and mixes into that
result instead of slicing and joining large temporary buffers. No GPU path is
included.

To reproduce:

```bash
pixi run bench
```

## How it works

`src/capi.mojo` is one compilation unit exporting a small C ABI. Python passes
buffer addresses, lengths, sample width, channel count, and scalar effect
parameters through `ctypes`; Python retains ownership of every allocation.
Mojo reconstructs typed pointers only after branching on the PCM width.

Samples are interleaved by frame (`L, R, L, R, ...`) in contiguous,
little-endian signed integer buffers. Gain uses pydub/audioop's exact floor
then saturate rule, mixing saturates at the destination width, and filters
maintain one recursive state per channel. The wrapper converts unsigned 8-bit
WAV bytes at the file boundary so all kernels see the same signed layout.

The build task emits `dist/libmojo-pydub.so`:

```bash
pixi run build
```
