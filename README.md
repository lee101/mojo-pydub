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
| apply_gain, 60 s stereo | 5.81 ms | 19.64 ms | 3.38x |
| normalize, 60 s stereo | 9.67 ms | 43.88 ms | 4.54x |
| overlay, 60 s + 30 s stereo | 6.89 ms | 25.20 ms | 3.66x |
| fade_in, 10 s stereo | 3.17 ms | 32.04 ms | 10.10x |
| pan, 60 s stereo | 15.35 ms | 130.78 ms | 8.52x |
| low_pass_filter, 10 s stereo | 3.01 ms | 349.39 ms | 115.88x |
| high_pass_filter, 10 s stereo | 3.53 ms | 782.96 ms | 221.68x |
| speedup 1.5x, 60 s stereo | 335.60 ms | 1254.22 ms | 3.74x |

Gain, mixing, and peak detection use float64-width SIMD with scalar remainder
handling. Gain and mixing buffers of at least 1,048,576 samples are split into
independent 262,144-sample chunks across at most eight CPU workers; smaller
inputs remain serial. The Python binding allocates the immutable result buffer
directly and fills it during the native call, avoiding an intermediate writable
buffer and final copy. Overlay copies the base once and mixes into that result
instead of slicing and joining large temporary buffers.

No GPU path is included. These kernels perform fewer than two arithmetic
operations per byte moved and are memory-bandwidth bound, so device transfer
and launch overhead would make a GPU path inappropriate.

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
