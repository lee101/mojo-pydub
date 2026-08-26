"""Integer PCM audio kernels exported through a stable C ABI."""

from max.algorithm import parallelize
from std.math import floor, sqrt
from std.sys.info import num_physical_cores, simd_width_of

comptime I8Ptr = UnsafePointer[Int8, AnyOrigin[mut=True]]
comptime I16Ptr = UnsafePointer[Int16, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 1_048_576
comptime PARALLEL_CHUNK = 262_144
comptime MAX_WORKERS = 8


def clip8(value: Int) -> Int8:
    return Int8(max(-128, min(127, value)))


def clip16(value: Int) -> Int16:
    return Int16(max(-32768, min(32767, value)))


def clip32(value: Int64) -> Int32:
    return Int32(max(Int64(-2147483648), min(Int64(2147483647), value)))


def audio_floor(value: Float64) -> Int64:
    return Int64(floor(value))


def gain8(src: I8Ptr, dst: I8Ptr, start: Int, end: Int, factor: Float64):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var values = floor(src.load[width=W](i).cast[DType.float64]() * factor)
        values = max(
            SIMD[DType.float64, W](-128.0),
            min(SIMD[DType.float64, W](127.0), values),
        )
        dst.store(i, values.cast[DType.int8]())
        i += W
    while i < end:
        dst[i] = clip8(Int(audio_floor(Float64(src[i]) * factor)))
        i += 1


def gain16(src: I16Ptr, dst: I16Ptr, start: Int, end: Int, factor: Float64):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var values = floor(src.load[width=W](i).cast[DType.float64]() * factor)
        values = max(
            SIMD[DType.float64, W](-32768.0),
            min(SIMD[DType.float64, W](32767.0), values),
        )
        dst.store(i, values.cast[DType.int16]())
        i += W
    while i < end:
        dst[i] = clip16(Int(audio_floor(Float64(src[i]) * factor)))
        i += 1


def gain32(src: I32Ptr, dst: I32Ptr, start: Int, end: Int, factor: Float64):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var values = floor(src.load[width=W](i).cast[DType.float64]() * factor)
        values = max(
            SIMD[DType.float64, W](-2147483648.0),
            min(SIMD[DType.float64, W](2147483647.0), values),
        )
        dst.store(i, values.cast[DType.int32]())
        i += W
    while i < end:
        dst[i] = clip32(audio_floor(Float64(src[i]) * factor))
        i += 1


@export("mpd_gain")
def mpd_gain(
    src_addr: Int, dst_addr: Int, n: Int, width: Int, factor: Float64
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0:
        return
    if width != 1 and width != 2 and width != 4:
        return
    @__parameter
    def worker(chunk: Int):
        var start = chunk * PARALLEL_CHUNK
        var end = min(n, start + PARALLEL_CHUNK)
        if width == 1:
            gain8(
                I8Ptr(unsafe_from_address=src_addr),
                I8Ptr(unsafe_from_address=dst_addr),
                start,
                end,
                factor,
            )
        elif width == 2:
            gain16(
                I16Ptr(unsafe_from_address=src_addr),
                I16Ptr(unsafe_from_address=dst_addr),
                start,
                end,
                factor,
            )
        else:
            gain32(
                I32Ptr(unsafe_from_address=src_addr),
                I32Ptr(unsafe_from_address=dst_addr),
                start,
                end,
                factor,
            )

    if n >= PARALLEL_THRESHOLD:
        var chunks = (n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK
        parallelize[worker](
            chunks, min(chunks, min(MAX_WORKERS, num_physical_cores()))
        )
    elif width == 1:
        gain8(
            I8Ptr(unsafe_from_address=src_addr),
            I8Ptr(unsafe_from_address=dst_addr),
            0,
            n,
            factor,
        )
    elif width == 2:
        gain16(
            I16Ptr(unsafe_from_address=src_addr),
            I16Ptr(unsafe_from_address=dst_addr),
            0,
            n,
            factor,
        )
    else:
        gain32(
            I32Ptr(unsafe_from_address=src_addr),
            I32Ptr(unsafe_from_address=dst_addr),
            0,
            n,
            factor,
        )


def mix8(
    a: I8Ptr,
    b: I8Ptr,
    dst: I8Ptr,
    start: Int,
    end: Int,
    a_factor: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var values = floor(a.load[width=W](i).cast[DType.float64]() * a_factor)
        values += b.load[width=W](i).cast[DType.float64]()
        values = max(
            SIMD[DType.float64, W](-128.0),
            min(SIMD[DType.float64, W](127.0), values),
        )
        dst.store(i, values.cast[DType.int8]())
        i += W
    while i < end:
        var av = audio_floor(Float64(a[i]) * a_factor)
        dst[i] = clip8(Int(av + Int64(b[i])))
        i += 1


def mix16(
    a: I16Ptr,
    b: I16Ptr,
    dst: I16Ptr,
    start: Int,
    end: Int,
    a_factor: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var values = floor(a.load[width=W](i).cast[DType.float64]() * a_factor)
        values += b.load[width=W](i).cast[DType.float64]()
        values = max(
            SIMD[DType.float64, W](-32768.0),
            min(SIMD[DType.float64, W](32767.0), values),
        )
        dst.store(i, values.cast[DType.int16]())
        i += W
    while i < end:
        var av = audio_floor(Float64(a[i]) * a_factor)
        dst[i] = clip16(Int(av + Int64(b[i])))
        i += 1


def mix32(
    a: I32Ptr,
    b: I32Ptr,
    dst: I32Ptr,
    start: Int,
    end: Int,
    a_factor: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var values = floor(a.load[width=W](i).cast[DType.float64]() * a_factor)
        values += b.load[width=W](i).cast[DType.float64]()
        values = max(
            SIMD[DType.float64, W](-2147483648.0),
            min(SIMD[DType.float64, W](2147483647.0), values),
        )
        dst.store(i, values.cast[DType.int32]())
        i += W
    while i < end:
        var av = audio_floor(Float64(a[i]) * a_factor)
        dst[i] = clip32(av + Int64(b[i]))
        i += 1


@export("mpd_mix")
def mpd_mix(
    a_addr: Int,
    b_addr: Int,
    dst_addr: Int,
    n: Int,
    width: Int,
    a_factor: Float64,
) abi("C"):
    if a_addr == 0 or b_addr == 0 or dst_addr == 0 or n <= 0:
        return
    if width != 1 and width != 2 and width != 4:
        return
    @__parameter
    def worker(chunk: Int):
        var start = chunk * PARALLEL_CHUNK
        var end = min(n, start + PARALLEL_CHUNK)
        if width == 1:
            mix8(
                I8Ptr(unsafe_from_address=a_addr),
                I8Ptr(unsafe_from_address=b_addr),
                I8Ptr(unsafe_from_address=dst_addr),
                start,
                end,
                a_factor,
            )
        elif width == 2:
            mix16(
                I16Ptr(unsafe_from_address=a_addr),
                I16Ptr(unsafe_from_address=b_addr),
                I16Ptr(unsafe_from_address=dst_addr),
                start,
                end,
                a_factor,
            )
        else:
            mix32(
                I32Ptr(unsafe_from_address=a_addr),
                I32Ptr(unsafe_from_address=b_addr),
                I32Ptr(unsafe_from_address=dst_addr),
                start,
                end,
                a_factor,
            )

    if n >= PARALLEL_THRESHOLD:
        var chunks = (n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK
        parallelize[worker](
            chunks, min(chunks, min(MAX_WORKERS, num_physical_cores()))
        )
    elif width == 1:
        mix8(
            I8Ptr(unsafe_from_address=a_addr),
            I8Ptr(unsafe_from_address=b_addr),
            I8Ptr(unsafe_from_address=dst_addr),
            0,
            n,
            a_factor,
        )
    elif width == 2:
        mix16(
            I16Ptr(unsafe_from_address=a_addr),
            I16Ptr(unsafe_from_address=b_addr),
            I16Ptr(unsafe_from_address=dst_addr),
            0,
            n,
            a_factor,
        )
    else:
        mix32(
            I32Ptr(unsafe_from_address=a_addr),
            I32Ptr(unsafe_from_address=b_addr),
            I32Ptr(unsafe_from_address=dst_addr),
            0,
            n,
            a_factor,
        )


def peak8(src: I8Ptr, n: Int) -> Int64:
    comptime W = simd_width_of[DType.float64]()
    var peak = Int64(0)
    var i = 0
    var vector_end = n - n % W
    while i < vector_end:
        var values = src.load[width=W](i).cast[DType.float64]()
        peak = max(peak, Int64(max(values, -values).reduce_max()))
        i += W
    while i < n:
        var value = Int64(src[i])
        peak = max(peak, -value if value < 0 else value)
        i += 1
    return peak


def peak16(src: I16Ptr, n: Int) -> Int64:
    comptime W = simd_width_of[DType.float64]()
    var peak = Int64(0)
    var i = 0
    var vector_end = n - n % W
    while i < vector_end:
        var values = src.load[width=W](i).cast[DType.float64]()
        peak = max(peak, Int64(max(values, -values).reduce_max()))
        i += W
    while i < n:
        var value = Int64(src[i])
        peak = max(peak, -value if value < 0 else value)
        i += 1
    return peak


def peak32(src: I32Ptr, n: Int) -> Int64:
    comptime W = simd_width_of[DType.float64]()
    var peak = Int64(0)
    var i = 0
    var vector_end = n - n % W
    while i < vector_end:
        var values = src.load[width=W](i).cast[DType.float64]()
        peak = max(peak, Int64(max(values, -values).reduce_max()))
        i += W
    while i < n:
        var value = Int64(src[i])
        peak = max(peak, -value if value < 0 else value)
        i += 1
    return peak


@export("mpd_peak")
def mpd_peak(src_addr: Int, n: Int, width: Int) abi("C") -> Int64:
    if src_addr == 0 or n <= 0:
        return 0
    if width == 1:
        return peak8(I8Ptr(unsafe_from_address=src_addr), n)
    if width == 2:
        return peak16(I16Ptr(unsafe_from_address=src_addr), n)
    if width == 4:
        return peak32(I32Ptr(unsafe_from_address=src_addr), n)
    return 0


@export("mpd_rms")
def mpd_rms(src_addr: Int, n: Int, width: Int) abi("C") -> Int64:
    if src_addr == 0 or n <= 0:
        return 0
    if width != 1 and width != 2 and width != 4:
        return 0
    var total = Float64(0.0)
    if width == 1:
        var src = I8Ptr(unsafe_from_address=src_addr)
        for i in range(n):
            var value = Float64(src[i])
            total += value * value
    elif width == 2:
        var src = I16Ptr(unsafe_from_address=src_addr)
        for i in range(n):
            var value = Float64(src[i])
            total += value * value
    else:
        var src = I32Ptr(unsafe_from_address=src_addr)
        for i in range(n):
            var value = Float64(src[i])
            total += value * value
    return Int64(floor(sqrt(total / Float64(n))))


@export("mpd_reverse")
def mpd_reverse(src_addr: Int, dst_addr: Int, n: Int, width: Int) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0:
        return
    if width != 1 and width != 2 and width != 4:
        return
    if width == 1:
        var src = I8Ptr(unsafe_from_address=src_addr)
        var dst = I8Ptr(unsafe_from_address=dst_addr)
        for i in range(n):
            dst[i] = src[n - i - 1]
    elif width == 2:
        var src = I16Ptr(unsafe_from_address=src_addr)
        var dst = I16Ptr(unsafe_from_address=dst_addr)
        for i in range(n):
            dst[i] = src[n - i - 1]
    else:
        var src = I32Ptr(unsafe_from_address=src_addr)
        var dst = I32Ptr(unsafe_from_address=dst_addr)
        for i in range(n):
            dst[i] = src[n - i - 1]


@export("mpd_fade")
def mpd_fade(
    src_addr: Int,
    dst_addr: Int,
    frames: Int,
    channels: Int,
    width: Int,
    from_factor: Float64,
    to_factor: Float64,
    frame_rate: Int,
    start_frame: Int,
    start_ms: Int,
    duration_ms: Int,
    coarse: Int,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or frames <= 0 or channels <= 0:
        return
    if width != 1 and width != 2 and width != 4:
        return
    if frame_rate <= 0 or duration_ms <= 0:
        return
    var delta = to_factor - from_factor
    var step = 0
    var next_boundary = Int(
        Float64(start_ms + 1) * Float64(frame_rate) / 1000.0
    )
    if width == 1:
        var src = I8Ptr(unsafe_from_address=src_addr)
        var dst = I8Ptr(unsafe_from_address=dst_addr)
        for frame in range(frames):
            if coarse:
                while start_frame + frame >= next_boundary:
                    step += 1
                    next_boundary = Int(
                        Float64(start_ms + step + 1)
                        * Float64(frame_rate)
                        / 1000.0
                    )
            var progress = (
                Float64(step)
                / Float64(duration_ms) if coarse else Float64(frame)
                / Float64(frames)
            )
            var factor = from_factor + delta * progress
            for channel in range(channels):
                var i = frame * channels + channel
                dst[i] = clip8(Int(audio_floor(Float64(src[i]) * factor)))
    elif width == 2:
        var src = I16Ptr(unsafe_from_address=src_addr)
        var dst = I16Ptr(unsafe_from_address=dst_addr)
        for frame in range(frames):
            if coarse:
                while start_frame + frame >= next_boundary:
                    step += 1
                    next_boundary = Int(
                        Float64(start_ms + step + 1)
                        * Float64(frame_rate)
                        / 1000.0
                    )
            var progress = (
                Float64(step)
                / Float64(duration_ms) if coarse else Float64(frame)
                / Float64(frames)
            )
            var factor = from_factor + delta * progress
            for channel in range(channels):
                var i = frame * channels + channel
                dst[i] = clip16(Int(audio_floor(Float64(src[i]) * factor)))
    else:
        var src = I32Ptr(unsafe_from_address=src_addr)
        var dst = I32Ptr(unsafe_from_address=dst_addr)
        for frame in range(frames):
            if coarse:
                while start_frame + frame >= next_boundary:
                    step += 1
                    next_boundary = Int(
                        Float64(start_ms + step + 1)
                        * Float64(frame_rate)
                        / 1000.0
                    )
            var progress = (
                Float64(step)
                / Float64(duration_ms) if coarse else Float64(frame)
                / Float64(frames)
            )
            var factor = from_factor + delta * progress
            for channel in range(channels):
                var i = frame * channels + channel
                dst[i] = clip32(audio_floor(Float64(src[i]) * factor))


@export("mpd_low_pass")
def mpd_low_pass(
    src_addr: Int,
    dst_addr: Int,
    frames: Int,
    channels: Int,
    width: Int,
    alpha: Float64,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or frames <= 0 or channels <= 0:
        return
    if width != 1 and width != 2 and width != 4:
        return
    if width == 1:
        var src = I8Ptr(unsafe_from_address=src_addr)
        var dst = I8Ptr(unsafe_from_address=dst_addr)
        for channel in range(channels):
            var last = Float64(src[channel])
            dst[channel] = src[channel]
            for frame in range(1, frames):
                var i = frame * channels + channel
                last += alpha * (Float64(src[i]) - last)
                dst[i] = Int8(Int(last))
    elif width == 2:
        var src = I16Ptr(unsafe_from_address=src_addr)
        var dst = I16Ptr(unsafe_from_address=dst_addr)
        for channel in range(channels):
            var last = Float64(src[channel])
            dst[channel] = src[channel]
            for frame in range(1, frames):
                var i = frame * channels + channel
                last += alpha * (Float64(src[i]) - last)
                dst[i] = Int16(Int(last))
    else:
        var src = I32Ptr(unsafe_from_address=src_addr)
        var dst = I32Ptr(unsafe_from_address=dst_addr)
        for channel in range(channels):
            var last = Float64(src[channel])
            dst[channel] = src[channel]
            for frame in range(1, frames):
                var i = frame * channels + channel
                last += alpha * (Float64(src[i]) - last)
                dst[i] = Int32(Int64(last))


@export("mpd_high_pass")
def mpd_high_pass(
    src_addr: Int,
    dst_addr: Int,
    frames: Int,
    channels: Int,
    width: Int,
    alpha: Float64,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or frames <= 0 or channels <= 0:
        return
    if width != 1 and width != 2 and width != 4:
        return
    if width == 1:
        var src = I8Ptr(unsafe_from_address=src_addr)
        var dst = I8Ptr(unsafe_from_address=dst_addr)
        for channel in range(channels):
            var last = Float64(src[channel])
            dst[channel] = src[channel]
            for frame in range(1, frames):
                var i = frame * channels + channel
                var prev = i - channels
                last = alpha * (last + Float64(src[i]) - Float64(src[prev]))
                dst[i] = clip8(Int(last))
    elif width == 2:
        var src = I16Ptr(unsafe_from_address=src_addr)
        var dst = I16Ptr(unsafe_from_address=dst_addr)
        for channel in range(channels):
            var last = Float64(src[channel])
            dst[channel] = src[channel]
            for frame in range(1, frames):
                var i = frame * channels + channel
                var prev = i - channels
                last = alpha * (last + Float64(src[i]) - Float64(src[prev]))
                dst[i] = clip16(Int(last))
    else:
        var src = I32Ptr(unsafe_from_address=src_addr)
        var dst = I32Ptr(unsafe_from_address=dst_addr)
        for channel in range(channels):
            var last = Float64(src[channel])
            dst[channel] = src[channel]
            for frame in range(1, frames):
                var i = frame * channels + channel
                var prev = i - channels
                last = alpha * (last + Float64(src[i]) - Float64(src[prev]))
                dst[i] = clip32(Int64(last))


@export("mpd_gain_stereo")
def mpd_gain_stereo(
    src_addr: Int,
    dst_addr: Int,
    frames: Int,
    channels: Int,
    width: Int,
    left_factor: Float64,
    right_factor: Float64,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or frames <= 0:
        return
    if channels != 1 and channels != 2:
        return
    if width != 1 and width != 2 and width != 4:
        return
    if width == 1:
        var src = I8Ptr(unsafe_from_address=src_addr)
        var dst = I8Ptr(unsafe_from_address=dst_addr)
        for frame in range(frames):
            var left = src[frame * channels]
            var right = left if channels == 1 else src[frame * channels + 1]
            dst[frame * 2] = clip8(
                Int(audio_floor(Float64(left) * left_factor))
            )
            dst[frame * 2 + 1] = clip8(
                Int(audio_floor(Float64(right) * right_factor))
            )
    elif width == 2:
        var src = I16Ptr(unsafe_from_address=src_addr)
        var dst = I16Ptr(unsafe_from_address=dst_addr)
        for frame in range(frames):
            var left = src[frame * channels]
            var right = left if channels == 1 else src[frame * channels + 1]
            dst[frame * 2] = clip16(
                Int(audio_floor(Float64(left) * left_factor))
            )
            dst[frame * 2 + 1] = clip16(
                Int(audio_floor(Float64(right) * right_factor))
            )
    else:
        var src = I32Ptr(unsafe_from_address=src_addr)
        var dst = I32Ptr(unsafe_from_address=dst_addr)
        for frame in range(frames):
            var left = src[frame * channels]
            var right = left if channels == 1 else src[frame * channels + 1]
            dst[frame * 2] = clip32(audio_floor(Float64(left) * left_factor))
            dst[frame * 2 + 1] = clip32(
                audio_floor(Float64(right) * right_factor)
            )
