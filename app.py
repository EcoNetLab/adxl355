import os
import time
import threading
import csv
import io

from datetime import datetime
from collections import deque
from flask import Flask, jsonify, render_template, send_file
from waitress import serve

# --------------------------------------------------
# Configuration
# --------------------------------------------------

import select
import numpy as np
from pathlib import Path

IIO_DEV = Path("/sys/bus/iio/devices/iio:device0")
IIO_NODE = "/dev/iio:device0"

SAMPLE_RATE = 1000
DISPLAY_RATE = 10
WINDOW_SECONDS = 10
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
DISPLAY_POINTS = 500          # points per axis sent to the browser
READ_SCANS = 100              # scans requested per read

SCAN_DTYPE = np.dtype([("xyz", ">i4", 3), ("pad", "V4"), ("ts", "<i8")])
SCAN_SIZE = SCAN_DTYPE.itemsize   # 24

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

app = Flask(__name__)

# Live graph buffer
timestamps = deque(maxlen=BUFFER_SIZE)
xs = deque(maxlen=BUFFER_SIZE)
ys = deque(maxlen=BUFFER_SIZE)
zs = deque(maxlen=BUFFER_SIZE)

stats = {"samples": 0, "dropped": 0, "started": None}

data_lock = threading.Lock()
stop_event = threading.Event()
rng = random.SystemRandom()

# Recording state
recording_file = None
csv_writer = None
recording_path = None
recording_started_at = None


# --------------------------------------------------
# Read data from buffer
# --------------------------------------------------

def decode(buf, scale, offset_ns):
    a = np.frombuffer(buf, dtype=SCAN_DTYPE)
    raw = (a["xyz"].astype(np.int32) >> 4) & 0xFFFFF
    raw = (raw ^ 0x80000) - 0x80000          # sign-extend from bit 19
    xyz = raw * scale                        # m/s²
    t_ns = a["ts"] + offset_ns               # monotonic -> wall clock
    return t_ns, xyz


# --------------------------------------------------
# Recording controls
# --------------------------------------------------

def get_recording_status():
    return {
        "recording": recording_file is not None,
        "filename": (
            os.path.basename(recording_path)
            if recording_path else None
        ),
        "started_at": (
            recording_started_at.isoformat(timespec="seconds")
            if recording_started_at else None
        ),
    }


@app.route("/api/recording/status")
def recording_status():
    with data_lock:
        return jsonify(get_recording_status())


@app.route("/api/recording/start", methods=["POST"])
def start_recording():
    global recording_file, csv_writer
    global recording_path, recording_started_at

    with data_lock:
        if recording_file is not None:
            return jsonify({
                "error": "Recording is already active.",
                **get_recording_status(),
            }), 409

        # Use the actual local start timestamp.
        recording_started_at = datetime.now().astimezone()

        filename = (
            "recording_"
            + recording_started_at.strftime("%Y-%m-%d_%H-%M-%S")
            + ".csv"
        )

        recording_path = os.path.join(RECORDINGS_DIR, filename)

        # Avoid overwriting an existing recording.
        suffix = 1
        while os.path.exists(recording_path):
            filename = (
                "recording_"
                + recording_started_at.strftime("%Y-%m-%d_%H-%M-%S")
                + f"_{suffix}.csv"
            )
            recording_path = os.path.join(RECORDINGS_DIR, filename)
            suffix += 1

        recording_file = open(
            recording_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        csv_writer = csv.writer(recording_file)
        csv_writer.writerow(["timestamp_utc", "x_m_s2", "y_m_s2", "z_m_s2"])
        recording_file.flush()

        return jsonify({
            "message": "Recording started.",
            **get_recording_status(),
        })


@app.route("/api/recording/stop", methods=["POST"])
def stop_recording():
    global recording_file, csv_writer

    with data_lock:
        if recording_file is None:
            return jsonify({
                "error": "No recording is active.",
                **get_recording_status(),
            }), 400

        recording_file.flush()
        os.fsync(recording_file.fileno())
        recording_file.close()

        recording_file = None
        csv_writer = None

        return jsonify({
            "message": "Recording stopped and saved.",
            **get_recording_status(),
        })


# --------------------------------------------------
# Acquisition thread: target 50 Hz
# --------------------------------------------------

def acquisition_loop():
    scale = float((IIO_DEV / "in_accel_scale").read_text())
    fd = os.open(IIO_NODE, os.O_RDONLY)

    # Assumes current_timestamp_clock is "monotonic" (the default).
    offset_ns = time.time_ns() - time.monotonic_ns()
    period_ns = 1e9 / SAMPLE_RATE

    pending = b""
    last_ts = None
    last_sync = time.monotonic()
    stats["started"] = time.monotonic()

    try:
        while not stop_event.is_set():
            # Wake up at least once a second so shutdown works.
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                continue

            pending += os.read(fd, SCAN_SIZE * READ_SCANS)
            n = len(pending) // SCAN_SIZE
            if n == 0:
                continue
            buf, pending = pending[:n * SCAN_SIZE], pending[n * SCAN_SIZE:]

            t_ns, xyz = decode(buf, scale, offset_ns)

            # Count missing samples from gaps in the timestamps.
            prev = t_ns[0] if last_ts is None else last_ts
            gaps = np.diff(t_ns, prepend=prev)
            missed = int(np.maximum(np.round(gaps / period_ns) - 1, 0).sum())
            last_ts = t_ns[-1]

            t_s = (t_ns / 1e9).tolist()
            x = xyz[:, 0].tolist()
            y = xyz[:, 1].tolist()
            z = xyz[:, 2].tolist()

            with data_lock:
                timestamps.extend(t_s)
                xs.extend(x)
                ys.extend(y)
                zs.extend(z)
                stats["samples"] += n
                stats["dropped"] += missed

                if recording_file is not None:
                    iso = np.datetime_as_string(
                        t_ns.astype("datetime64[ns]"),
                        unit="ms", timezone="UTC",
                    )
                    csv_writer.writerows(zip(iso, x, y, z))
                    recording_file.flush()

                    if time.monotonic() - last_sync >= 1.0:
                        os.fsync(recording_file.fileno())
                        last_sync = time.monotonic()
    finally:
        os.close(fd)


# --------------------------------------------------
# Web interface and graph API
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
G = 9.80665

def analyse_axis(sig, fs):
    """sig in m/s², fs in Hz. Returns None if there is too little data."""
    sig = sig - sig.mean()                       # remove DC (gravity/offset)
    peak_ms2 = float((sig.max() - sig.min()) / 2)
    result = {
        "frequency": None,
        "peak_g": peak_ms2 / G,
        "peak_ms2": peak_ms2,
        "disp_mm": None,
    }

    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    valid = freqs >= 0.5                         # ignore drift below 0.5 Hz
    if not valid.any():
        return result

    k = int(np.argmax(np.where(valid, spec, 0)))
    if 0 < k < len(spec) - 1 and spec[k] > 0:    # parabolic peak interpolation
        a, b, c = (np.log(spec[k + i] + 1e-12) for i in (-1, 0, 1))
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    f = float(k * fs / len(sig))

    if f > 0:
        result["frequency"] = f
        result["disp_mm"] = peak_ms2 / (2 * np.pi * f) ** 2 * 1000
    return result

def get_data():
    with data_lock:
        t = np.array(timestamps)
        x = np.array(xs)
        y = np.array(ys)
        z = np.array(zs)

    mask = t >= time.time() - WINDOW_SECONDS
    t, x, y, z = t[mask], x[mask], y[mask], z[mask]

    analysis = None
    if len(t) > 256 and t[-1] > t[0]:
        fs = (len(t) - 1) / (t[-1] - t[0])       # measured, not assumed
        analysis = {
            "x": analyse_axis(x, fs),
            "y": analyse_axis(y, fs),
            "z": analyse_axis(z, fs),
        }

    step = max(1, len(t) // DISPLAY_POINTS)
    return jsonify({
        "timestamps": t[::step].tolist(),
        "x": x[::step].tolist(),
        "y": y[::step].tolist(),
        "z": z[::step].tolist(),
        "analysis": analysis,
        "sample_rate": SAMPLE_RATE,
        "display_rate": DISPLAY_RATE,
    })


@app.route("/api/stats")
def get_stats():
    with data_lock:
        elapsed = time.monotonic() - stats["started"] if stats["started"] else 0
        return jsonify({
            "samples": stats["samples"],
            "dropped": stats["dropped"],
            "measured_rate_hz": stats["samples"] / elapsed if elapsed else 0,
        })


# --------------------------------------------------
# Download current or most recent recording
# --------------------------------------------------

@app.route("/download")
def download_recording():
    with data_lock:
        if recording_path is None:
            return jsonify({
                "error": "Start a recording before downloading."
            }), 400

        # Include all samples written up to this moment.
        if recording_file is not None:
            recording_file.flush()

        with open(recording_path, "rb") as source:
            snapshot = io.BytesIO(source.read())

        filename = os.path.basename(recording_path)

    snapshot.seek(0)

    return send_file(
        snapshot,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    acquisition_thread = threading.Thread(
        target=acquisition_loop,
        daemon=True,
    )

    acquisition_thread.start()

    print("Starting Device Viewer...")
    print("Listening on http://0.0.0.0:8080")

    try:
        serve(
            app,
            host="0.0.0.0",
            port=8080,
            threads=4,
        )

    except KeyboardInterrupt:
        print("Shutting down Device Viewer...")

    finally:
        stop_event.set()
        acquisition_thread.join(timeout=2)

        with data_lock:
            if recording_file is not None:
                recording_file.flush()
                os.fsync(recording_file.fileno())
                recording_file.close()