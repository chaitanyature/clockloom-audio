import json
import os
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape

from scipy.io import wavfile

from .alignment import estimate
from .audio import read_wav, render


def plot(path, reports):
    """Static diagnostic plot, scaled in milliseconds around each fitted clock."""
    w = 960
    panel = 180
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{60 + panel * len(reports)}" viewBox="0 0 {w} {60 + panel * len(reports)}">',
        '<rect width="100%" height="100%" fill="#122c32"/>',
        '<g font-family="sans-serif" fill="#e9f2ed"><text x="32" y="37" font-size="23">Clock fit · landmark residuals</text>',
    ]
    for i, r in enumerate(reports):
        top = 60 + i * panel
        left, right = 92, 930
        middle = top + 89
        rows = [s for s in r["landmarks"] if s["status"] in ("inlier", "outlier")]
        limit = max(
            0.35,
            max(abs(p["residual_ms"]) for p in rows if p["status"] == "inlier") * 1.3,
        )
        width = max(s["source_s"] for s in r["landmarks"])
        parts.extend(
            [
                f'<text x="32" y="{top + 21}" font-size="16">{escape(r["source"])} · {r["drift_ppm"]:+.1f} ppm · offset {r["offset_s"]:+.4f} s</text>',
                f'<path d="M{left} {middle}H{right}" stroke="#6c8f91" stroke-dasharray="4 4"/>',
            ]
        )
        for mark in (-limit, 0, limit):
            y = middle - mark / limit * 40
            parts.append(
                f'<text x="79" y="{y + 4}" font-size="11" fill="#b5cccb" text-anchor="end">{mark:+.2f} ms</text>'
            )
        for p in r["landmarks"]:
            x = left + (right - left) * p["source_s"] / width
            if p["status"] == "inlier":
                y = middle - 40 * p["residual_ms"] / limit
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="#6eddb2"/>'
                )
            else:
                y = middle + 47
                parts.append(
                    f'<path d="M{x - 3:.2f} {y - 3}l6 6m0 -6l-6 6" stroke="#eeb876"/>'
                )
        parts.append(
            f'<text x="{left}" y="{middle + 70}" fill="#b5cccb" font-size="12">0 s</text>'
        )
        parts.append(
            f'<text x="{right}" y="{middle + 70}" text-anchor="end" fill="#b5cccb" font-size="12">{width:.2f} s · recorder time</text>'
        )
    path.write_text("\n".join(parts + ["</g></svg>"]), encoding="utf-8")


def align_files(ref_path, track_paths, out, *, anchors=31, max_ppm=1800):
    ref_path = Path(ref_path)
    out = Path(out)
    if not track_paths:
        raise ValueError("supply at least one track")
    if out.exists():
        raise ValueError(f"output directory exists: {out}")
    paths = [Path(p) for p in track_paths]
    if len({p.resolve() for p in [ref_path, *paths]}) != len(paths) + 1:
        raise ValueError("reference and track files must be distinct")
    rate, reference = read_wav(ref_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": 1,
        "reference": ref_path.name,
        "reference_rate": rate,
        "frames": len(reference),
        "tracks": [],
    }
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".clockloom-") as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        for index, path in enumerate(paths, 1):
            fs, source = read_wav(path)
            clock, track = estimate(
                reference, rate, source, fs, anchors=anchors, max_ppm=max_ppm
            )
            audio, (first, last) = render(source, fs, rate, len(reference), clock)
            name = f"{index:02d}-{path.stem}.aligned.wav"
            wavfile.write(bundle / name, rate, audio)
            track.update(
                source=path.name,
                output=name,
                channels=source.shape[1],
                native_rate=fs,
                available_frames=[first, last],
            )
            report["tracks"].append(track)
        (bundle / "alignment.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        plot(bundle / "residuals.svg", report["tracks"])
        os.rename(bundle, out)
    return report
