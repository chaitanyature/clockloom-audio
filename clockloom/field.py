"""Generate an original time-varying pier soundscape and two independent recorder clocks."""

from pathlib import Path
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


def scene(rate, seconds, seed=41):
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    t = np.arange(n) / rate
    wind = sosfilt(butter(3, 180, fs=rate, output="sos"), rng.normal(size=n))
    water = sosfilt(
        butter(2, [350, 2000], btype="band", fs=rate, output="sos"), rng.normal(size=n)
    )
    amp = 0.11 + 0.07 * np.sin(t * 1.67 + 0.3 * np.sin(t))
    out = 0.35 * wind + amp * water
    for when in np.arange(0.35, seconds - 0.2, 0.81):
        j = round((when + rng.uniform(-0.09, 0.09)) * rate)
        k = min(n - j, round(0.16 * rate))
        u = np.arange(k) / rate
        freq = rng.uniform(700, 2100)
        out[j : j + k] += (
            rng.uniform(0.18, 0.4)
            * np.exp(-u * 29)
            * np.sin(2 * np.pi * (freq * u + 950 * u * u))
        )
    return 0.65 * out / max(np.max(np.abs(out)), 1e-8)


def draw_recording(
    reference, rate, device_rate, offset, drift_ppm, duration, rng, *, noise=0.008
):
    # Independent data generator: forward time map + linear interpolation.
    time = (
        offset
        + (1 + drift_ppm * 1e-6)
        * np.arange(round(duration * device_rate))
        / device_rate
    )
    clean = np.interp(
        time, np.arange(len(reference)) / rate, reference, left=0, right=0
    )
    return clean * 0.79 + noise * rng.normal(size=len(time))


def generate(directory):
    root = Path(directory)
    root.mkdir()
    rate = 16000
    reference = scene(rate, 14)
    rng = np.random.default_rng(1984)

    def write(path, fs, samples):
        wavfile.write(
            path, fs, np.round(np.clip(samples, -0.999, 0.999) * 32767).astype(np.int16)
        )

    write(root / "pier.wav", rate, reference)
    truth = []
    for name, fs, offset, drift, length in [
        ("quay", 12000, 0.93, 460.0, 11.7),
        ("boat", 22050, -0.46, -680.0, 13.8),
    ]:
        track = draw_recording(reference, rate, fs, offset, drift, length, rng)
        if name == "boat":
            # An intermittent cable disconnect; the timeline keeps its samples.
            track[fs * 6 : fs * 8] = rng.normal(0, 0.014, fs * 2)
            track = np.stack((track, -0.55 * track), axis=1)
        write(root / (name + ".wav"), fs, track)
        truth.append({"file": name + ".wav", "offset_s": offset, "drift_ppm": drift})
    (root / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return root, truth
