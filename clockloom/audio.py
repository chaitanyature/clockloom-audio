from math import gcd
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_wav(path):
    rate, pcm = wavfile.read(Path(path))
    if rate < 100 or pcm.ndim not in (1, 2) or not len(pcm):
        raise ValueError(
            "expected a nonempty mono or multichannel WAV at 100 Hz or higher"
        )
    if np.issubdtype(pcm.dtype, np.signedinteger):
        depth = 8 * pcm.dtype.itemsize
        pcm = pcm.astype(np.float64) / (2.0 ** (depth - 1))
    elif pcm.dtype == np.uint8:
        pcm = (pcm.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(pcm.dtype, np.floating):
        pcm = pcm.astype(np.float64)
    else:
        raise ValueError("unsupported sample representation")
    if not np.isfinite(pcm).all():
        raise ValueError("nonfinite PCM samples")
    return int(rate), pcm.reshape(len(pcm), -1)


def nominal_resample(pcm, source, target):
    if source == target:
        return pcm
    d = gcd(source, target)
    return resample_poly(pcm, target // d, source // d, axis=0, padtype="constant")


def render(pcm, source_rate, reference_rate, frames, clock):
    """Sinc-interpolate source positions after nominal sample-rate conversion.

    source seconds u map to reference seconds offset + scale*u. Output support
    is half-open and branch-free for the caller: unavailable frames are zeros.
    """
    audio = nominal_resample(pcm, source_rate, reference_rate)
    audio = np.asarray(audio, dtype=float).reshape(len(audio), -1)
    output = np.zeros((frames, audio.shape[1]), dtype=np.float32)
    begin = int(np.clip(np.ceil(clock.offset * reference_rate), 0, frames))
    stop = int(
        np.clip(
            np.ceil(
                (clock.offset + clock.scale * len(pcm) / source_rate) * reference_rate
            ),
            begin,
            frames,
        )
    )
    radius = 16
    taps = np.arange(-radius + 1, radius + 1)
    # A smooth cutoff accounts for the remaining affine downsampling after
    # polyphase conversion to the nominal output sample rate.
    cutoff = min(1.0, clock.scale)
    for start in range(begin, stop, 16384):
        end = min(stop, start + 16384)
        pos = (np.arange(start, end) - clock.offset * reference_rate) / clock.scale
        ix = np.floor(pos).astype(np.int64)[:, None] + taps
        dist = pos[:, None] - ix
        weights = np.sinc(cutoff * dist) * np.sinc(dist / radius) * cutoff
        weights[(ix < 0) | (ix >= len(audio)) | (np.abs(dist) >= radius)] = 0
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        block = audio[np.clip(ix, 0, len(audio) - 1)]
        output[start:end] = np.einsum("ij,ijk->ik", weights, block)
    return output, (begin, stop)
