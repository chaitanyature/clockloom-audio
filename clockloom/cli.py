import argparse
import json
from pathlib import Path
import sys
import tempfile
from .bundle import align_files
from .field import generate


def demo(directory):
    out = Path(directory)
    if out.exists():
        raise ValueError(f"demo directory exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".demo-") as tmp:
        demo_root = Path(tmp) / "demo"
        demo_root.mkdir()
        source, truth = generate(demo_root / "inputs")
        report = align_files(
            source / "pier.wav",
            [source / x["file"] for x in truth],
            demo_root / "alignment",
        )
        errors = []
        for fit, known in zip(report["tracks"], truth):
            dt = (fit["offset_s"] - known["offset_s"]) * 1000
            ppm = fit["drift_ppm"] - known["drift_ppm"]
            if abs(dt) > 1 or abs(ppm) > 35:
                raise ValueError(
                    f"generated clock validation failed for {fit['source']}"
                )
            errors.append(
                {
                    "file": fit["source"],
                    "offset_error_ms": dt,
                    "drift_error_ppm": ppm,
                    "inliers": fit["quality"]["inliers"],
                }
            )
        (demo_root / "validation.json").write_text(json.dumps(errors, indent=2) + "\n")
        demo_root.rename(out)
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="clockloom",
        description="Align independent WAV recorders to a reference clock",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "demo",
        help="generate a known scene and validate two independent clock estimates",
    )
    d.add_argument("directory", type=Path)
    a = sub.add_parser("align", help="align and render one or more WAVs")
    a.add_argument("reference", type=Path)
    a.add_argument("tracks", nargs="+", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--anchors", type=int, default=31)
    a.add_argument("--max-ppm", type=float, default=1800)
    args = p.parse_args(argv)
    try:
        if args.cmd == "demo":
            summary = demo(args.directory)
        else:
            report = align_files(
                args.reference,
                args.tracks,
                args.out,
                anchors=args.anchors,
                max_ppm=args.max_ppm,
            )
            summary = [
                {
                    k: r[k]
                    for k in [
                        "source",
                        "offset_s",
                        "drift_ppm",
                        "quality",
                        "available_frames",
                    ]
                }
                for r in report["tracks"]
            ]
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"clockloom: {exc}", file=sys.stderr)
        return 2
