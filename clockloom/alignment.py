"""Normalized waveform correlation followed by a robust affine clock fit."""

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.signal import correlate

from .audio import nominal_resample


class AlignmentError(ValueError):
    """The recording does not support a unique clock within the declared search bounds."""


@dataclass(frozen=True)
class Clock:
    offset: float
    scale: float

    @property
    def drift_ppm(self):
        return (self.scale - 1.0) * 1e6


def analysis_channel(pcm, rate, analysis_rate):
    pcm = np.asarray(pcm)
    if pcm.ndim == 2:
        # Retain one physical channel. Averaging antiphase stereo would cancel transients.
        pcm = pcm[:, np.argmax(np.var(pcm, axis=0))]
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise AlignmentError("analysis requires finite PCM samples")
    x = np.ascontiguousarray(
        nominal_resample(pcm, rate, analysis_rate), dtype=np.float64
    )
    return np.diff(x, prepend=x[0])


def evidence(reference, recording, rate, window, anchors, min_correlation, min_gap):
    size = int(round(window * rate))
    if size < 32 or size > min(len(reference), len(recording)):
        raise AlignmentError("recording shorter than the requested landmark window")
    partial = np.concatenate(([0.0], np.cumsum(reference)))
    square = np.concatenate(([0.0], np.cumsum(reference * reference)))
    sums = partial[size:] - partial[:-size]
    power = np.maximum(0, square[size:] - square[:-size] - sums * sums / size)
    samples = []
    for start in np.unique(np.linspace(0, len(recording) - size, anchors, dtype=int)):
        template = recording[start : start + size]
        centered = template - template.mean()
        energy = float(np.dot(centered, centered))
        row = {
            "source_s": float((start + (size - 1) / 2) / rate),
            "reference_s": None,
            "correlation": 0.0,
            "separation": 0.0,
            "status": "quiet",
        }
        samples.append(row)
        if energy < 1e-12:
            continue
        num = correlate(reference, centered, mode="valid", method="fft")
        den = np.sqrt(power * energy)
        score = np.zeros_like(num)
        np.divide(num, den, out=score, where=den > 1e-10)
        # A polarity reversal between microphone chains still provides a timing landmark.
        absolute = np.clip(np.abs(score), 0, 1)
        idx = int(np.argmax(absolute))
        peak = float(absolute[idx])
        guard = max(3, int(0.018 * rate))
        runner = max(
            float(absolute[: max(0, idx - guard)].max(initial=0)),
            float(absolute[idx + guard + 1 :].max(initial=0)),
        )
        sub = 0.0
        if 0 < idx < len(absolute) - 1:
            left, mid, right = absolute[idx - 1 : idx + 2]
            curvature = left - 2 * mid + right
            if curvature < -1e-12:
                sub = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
        row.update(
            reference_s=float((idx + sub + (size - 1) / 2) / rate),
            correlation=peak,
            separation=peak - runner,
            status="candidate"
            if peak >= min_correlation and peak - runner >= min_gap
            else "ambiguous",
        )
    return samples


def fit(rows, rate, max_ppm, min_span):
    usable = [r for r in rows if r["status"] == "candidate"]
    if len(usable) < 4:
        raise AlignmentError("fewer than four distinct, sufficiently strong landmarks")
    x = np.array([r["source_s"] for r in usable])
    y = np.array([r["reference_s"] for r in usable])
    tolerance = max(2.5 / rate, 0.001)
    best = None
    mask = None
    # Deterministic two-point consensus; no random seed changes the selected clock.
    for i, j in combinations(range(len(x)), 2):
        if x[j] - x[i] < min_span:
            continue
        slope = (y[j] - y[i]) / (x[j] - x[i])
        if abs(slope - 1) > max_ppm * 1e-6 + 2 * tolerance / (x[j] - x[i]):
            continue
        residual = np.abs(y - (y[i] + slope * (x - x[i])))
        hits = residual <= tolerance
        rank = (int(hits.sum()), -float(np.median(residual[hits])))
        if best is None or rank > best:
            best, mask = rank, hits
    if (
        mask is None
        or mask.sum() < max(4, int(np.ceil(0.65 * len(x))))
        or np.ptp(x[mask]) < min_span
    ):
        raise AlignmentError("landmarks do not agree on one bounded affine clock")

    def regress(selected):
        center = x[selected].mean()
        coef = np.polyfit(x[selected] - center, y[selected], 1)
        return float(coef[1] - coef[0] * center), float(coef[0])

    offset, slope = regress(mask)
    mask = np.abs(y - offset - slope * x) <= tolerance
    if mask.sum() < max(4, int(np.ceil(0.65 * len(x)))) or np.ptp(x[mask]) < min_span:
        raise AlignmentError("insufficient inlier coverage after refinement")
    offset, slope = regress(mask)
    if abs(slope - 1) > max_ppm * 1e-6:
        raise AlignmentError("estimated drift exceeds the declared ppm limit")
    for r, used in zip(usable, mask):
        r["status"] = "inlier" if used else "outlier"
        r["residual_ms"] = float(
            (r["reference_s"] - offset - slope * r["source_s"]) * 1000
        )
    residual = y[mask] - offset - slope * x[mask]
    return Clock(offset, slope), {
        "inliers": int(mask.sum()),
        "candidates": len(usable),
        "inlier_span_s": float(np.ptp(x[mask])),
        "rms_ms": float(np.sqrt(np.mean(residual**2)) * 1000),
        "max_abs_ms": float(np.max(np.abs(residual)) * 1000),
    }


def estimate(
    reference,
    reference_rate,
    recording,
    recording_rate,
    *,
    analysis_rate=6000,
    window=0.18,
    anchors=31,
    max_ppm=1800,
    min_correlation=0.38,
    min_gap=0.1,
):
    if (
        not 4 <= anchors <= 100
        or not 200 <= analysis_rate <= 48000
        or not np.isfinite(window)
        or window <= 0
        or not 0 < max_ppm < 1e5
    ):
        raise ValueError("invalid landmark or clock settings")
    rate = min(analysis_rate, reference_rate, recording_rate)
    ref = analysis_channel(reference, reference_rate, rate)
    rec = analysis_channel(recording, recording_rate, rate)
    rows = evidence(ref, rec, rate, window, anchors, min_correlation, min_gap)
    clock, quality = fit(rows, rate, max_ppm, max(0.55, window * 3))
    return clock, {
        "offset_s": clock.offset,
        "scale": clock.scale,
        "drift_ppm": clock.drift_ppm,
        "analysis_hz": rate,
        "quality": quality,
        "landmarks": rows,
    }
