import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from clockloom import AlignmentError, Clock, estimate, read_wav, render
from clockloom.alignment import fit
from clockloom.bundle import align_files
from clockloom.field import draw_recording, scene

ROOT = Path(__file__).resolve().parents[1]


class ClockTests(unittest.TestCase):
    def test_recovers_both_clock_directions(self):
        rate = 10000
        ref = scene(rate, 7)
        for n, (native, delta, ppm) in enumerate(
            [(11025, 0.42, -580), (8000, -0.18, 650)]
        ):
            rng = np.random.default_rng(21 + n)
            track = draw_recording(ref, rate, native, delta, ppm, 6.5, rng)
            clock, report = estimate(ref, rate, track, native, anchors=23)
            self.assertAlmostEqual(clock.offset, delta, delta=0.0005)
            self.assertAlmostEqual(clock.drift_ppm, ppm, delta=18)
            self.assertGreater(report["quality"]["inlier_span_s"], 4)
            self.assertEqual(
                report["quality"]["inliers"],
                sum(r["status"] == "inlier" for r in report["landmarks"]),
            )

    def test_held_out_waveform_improves(self):
        rate = 10000
        ref = scene(rate, 8, seed=997)
        known = Clock(0.31, 1.0011)
        track = draw_recording(
            ref,
            rate,
            rate,
            known.offset,
            known.drift_ppm,
            7,
            np.random.default_rng(9),
            noise=0.001,
        )
        estimate_clock, _ = estimate(ref, rate, track, rate, anchors=17)
        fixed, (first, last) = render(
            track[:, None], rate, rate, len(ref), estimate_clock
        )
        naive, _ = render(track[:, None], rate, rate, len(ref), Clock(known.offset, 1))
        sl = slice(5 * rate, 6 * rate)

        def corr(a, b):
            return np.corrcoef(a, b)[0, 1]

        self.assertGreater(corr(fixed[sl, 0], ref[sl]), 0.96)
        self.assertGreater(
            corr(fixed[sl, 0], ref[sl]) - corr(naive[sl, 0], ref[sl]), 0.3
        )
        self.assertFalse(fixed[:first].any())
        self.assertFalse(fixed[last:].any())

    def test_inverted_stereo_and_unmatched_dropout(self):
        fs = 8000
        ref = scene(fs, 7)
        track = draw_recording(ref, fs, fs, 0.2, 300, 6.4, np.random.default_rng(3))
        track[fs * 2 : fs * 4] = 0
        stereo = np.stack([-track, 0.4 * track], axis=1)
        clock, r = estimate(ref, fs, stereo, fs)
        self.assertAlmostEqual(clock.offset, 0.2, delta=0.0006)
        self.assertTrue(any(p["status"] == "quiet" for p in r["landmarks"]))
        audio, _ = render(stereo, fs, fs, len(ref), clock)
        self.assertEqual(audio.shape, (len(ref), 2))
        np.testing.assert_allclose(audio[:, 1], -0.4 * audio[:, 0], atol=1e-7)

    def test_ambiguity_silence_unrelated_or_short_rejected(self):
        rng = np.random.default_rng(13)
        fs = 4000
        ref = rng.normal(size=6 * fs)
        loop = np.tile(rng.normal(size=fs), 6)
        for reference, track in [
            (ref, np.zeros(3 * fs)),
            (ref, rng.normal(size=3 * fs)),
            (loop, loop[fs // 2 : 4 * fs]),
            (ref, ref[:12]),
        ]:
            with self.subTest(size=len(track)), self.assertRaises(AlignmentError):
                estimate(reference, fs, track, fs)

    def test_consensus_ignores_plausible_false_matches(self):
        x = np.linspace(0, 13, 24)
        y = 0.19 + 1.00032 * x
        rows = [
            {"source_s": float(a), "reference_s": float(b), "status": "candidate"}
            for a, b in zip(x, y)
        ]
        for i in (3, 6, 7, 13, 22):
            rows[i]["reference_s"] += 0.6 + i * 0.01
        clock, stats = fit(rows, 6000, 800, 0.55)
        self.assertAlmostEqual(clock.offset, 0.19)
        self.assertAlmostEqual(clock.drift_ppm, 320)
        self.assertEqual(stats["inliers"], 19)
        self.assertEqual(sum(s["status"] == "outlier" for s in rows), 5)

    def test_clock_bound_is_enforced(self):
        x = np.arange(10.0)
        rows = [
            {"source_s": s, "reference_s": 0.4 + 1.008 * s, "status": "candidate"}
            for s in x
        ]
        with self.assertRaises(AlignmentError):
            fit(rows, 6000, 100, 0.5)

    def test_no_coverage_and_identity(self):
        rng = np.random.default_rng(4)
        track = rng.normal(size=(512, 2))
        same, support = render(track, 8000, 8000, 512, Clock(0, 1))
        self.assertEqual(support, (0, 512))
        np.testing.assert_allclose(same, track, rtol=1e-6, atol=1e-6)
        silent, support = render(track, 8000, 8000, 512, Clock(2, 1))
        self.assertEqual(support, (512, 512))
        self.assertFalse(silent.any())

    def test_bundle_failure_leaves_no_partial_products(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            x = scene(5000, 5)
            np.random.default_rng(5)
            for n, a in [("ref", x), ("valid", x[500:]), ("quiet", np.zeros_like(x))]:
                wavfile.write(p / (n + ".wav"), 5000, (a * 20000).astype(np.int16))
            with self.assertRaises(AlignmentError):
                align_files(
                    p / "ref.wav", [p / "valid.wav", p / "quiet.wav"], p / "output"
                )
            self.assertFalse((p / "output").exists())
            self.assertFalse(list(p.glob(".clockloom-*")))
            with self.assertRaises(ValueError):
                align_files(p / "ref.wav", [p / "ref.wav"], p / "output")

    def test_cli_renders_and_preserves_source_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "site"
            cmd = [sys.executable, "-m", "clockloom", "demo", str(p)]
            errors = json.loads(subprocess.check_output(cmd, cwd=ROOT, text=True))
            self.assertTrue(all(abs(e["drift_error_ppm"]) < 35 for e in errors))
            report = json.loads((p / "alignment/alignment.json").read_text())
            fs, pcm = read_wav(p / "alignment" / report["tracks"][1]["output"])
            self.assertEqual((fs, pcm.shape), (16000, (224000, 2)))
            self.assertIn("<svg", (p / "alignment/residuals.svg").read_text())
            self.assertNotEqual(
                subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode, 0
            )
