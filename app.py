import os
import time
import select
import threading
import csv

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, send_file
from waitress import serve

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IIO_DEV = Path("/sys/bus/iio/devices/iio:device0")
IIO_NODE = "/dev/iio:device0"

SAMPLE_RATE = 1000          # must match the sensor's sampling_frequency
DISPLAY_RATE = 10           # browser refresh rate (Hz), informational
WINDOW_SECONDS = 10
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
DISPLAY_POINTS = 500        # points per axis sent to the browser
READ_SCANS = 100            # scans requested per read()

G = 9.80665

# One IIO scan: x, y, z (big-endian, 20 bits in 32, shifted by 4),
# 4 bytes padding, then a 64-bit little-endian timestamp in ns.
SCAN_DTYPE = np.dtype([("xyz", ">i4", 3), ("pad", "V4"), ("ts", "<i8")])
SCAN_SIZE = SCAN_DTYPE.itemsize          # 24 bytes
assert SCAN_SIZE == 24

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

app = Flask(__name__)

# Live graph buffers
timestamps = deque(maxlen=BUFFER_SIZE)
xs = deque(maxlen=BUFFER_SIZE)
ys = deque(maxlen=BUFFER_SIZE)
zs = deque(maxlen=BUFFER_SIZE)

stats = {
    "samples": 0,
    "dropped": 0,
    "started": None,
    "error": None,
}

data_lock = threading.Lock()
stop_event = threading.Event()

# Recording state
recording_file = None
csv_writer = None
recording_path = None
recording_started_at = None


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

        recording_started_at = datetime.now().astimezone()

        stem = (
            "recording_"
            + recording_started_at.strftime("%Y-%m-%d_%H-%M-%S")
        )
        filename = stem + ".csv"
        recording_path = os.path.join(RECORDINGS_DIR, filename)

        # Avoid overwriting an existing recording.
        suffix = 1
        while os.path.exists(recording_path):
            filename = f"{stem}_{suffix}.csv"
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
# ADXL355 acquisition (IIO buffer, DRDY-triggered)
# --------------------------------------------------

def clock_offset_ns():
    """Offset to add to IIO timestamps to get Unix time in ns."""
    try:
        clock = (IIO_DEV / "current_timestamp_clock").read_text().strip()
    except OSError:
        clock = "monotonic"

    if clock == "realtime":
        return 0, clock
    if clock == "boottime":
        return time.time_ns() - time.clock_gettime_ns(time.CLOCK_BOOTTIME), clock
    return time.time_ns() - time.monotonic_ns(), clock


def decode(buf, scale, offset_ns):
    a = np.frombuffer(buf, dtype=SCAN_DTYPE)
    raw = (a["xyz"].astype(np.int32) >> 4) & 0xFFFFF
    raw = (raw ^ 0x80000) - 0x80000          # sign-extend from bit 19
    xyz = raw * scale                        # m/s²
    t_ns = a["ts"] + offset_ns               # kernel clock -> Unix time
    return t_ns, xyz


def acquisition_loop():
    global csv_writer

    try:
        scale = float((IIO_DEV / "in_accel_scale").read_text())
        fd = os.open(IIO_NODE, os.O_RDONLY)
    except OSError as exc:
        with data_lock:
            stats["error"] = f"Cannot open ADXL355: {exc}"
        return

    offset_ns, clock = clock_offset_ns()
    period_ns = 1e9 / SAMPLE_RATE

    pending = b""
    last_ts = None
    last_sync = time.monotonic()
    stats["started"] = time.monotonic()
    print(f"ADXL355 opened (timestamp clock: {clock}, scale: {scale})")

    try:
        while not stop_event.is_set():
            # Wake at least once a second so shutdown stays responsive.
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                continue

            pending += os.read(fd, SCAN_SIZE * READ_SCANS)
            n = len(pending) // SCAN_SIZE
            if n == 0:
                continue

            buf = pending[:n * SCAN_SIZE]
            pending = pending[n * SCAN_SIZE:]

            t_ns, xyz = decode(buf, scale, offset_ns)

            # Count missing samples from gaps in the timestamps.
            prev = t_ns[0] if last_ts is None else last_ts
            gaps = np.diff(t_ns, prepend=prev)
            missed = int(
                np.maximum(np.round(gaps / period_ns) - 1, 0).sum()
            )
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
                        unit="ms",
                        timezone="UTC",
                    )
                    csv_writer.writerows(zip(iso, x, y, z))
                    recording_file.flush()

                    # Durable disk sync about once per second.
                    if time.monotonic() - last_sync >= 1.0:
                        os.fsync(recording_file.fileno())
                        last_sync = time.monotonic()
    except Exception as exc:
        with data_lock:
            stats["error"] = f"Acquisition stopped: {exc}"
        raise
    finally:
        os.close(fd)


# --------------------------------------------------
# Signal analysis (runs on the full-rate data)
# --------------------------------------------------

def analyse_axis(sig, fs):
    """sig in m/s², fs in Hz."""
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
    kf = float(k)
    if 0 < k < len(spec) - 1 and spec[k] > 0:    # parabolic interpolation
        a, b, c = (np.log(spec[k + i] + 1e-12) for i in (-1, 0, 1))
        denom = a - 2 * b + c
        if denom != 0:
            kf = k + 0.5 * (a - c) / denom
    f = kf * fs / len(sig)

    if f > 0:
        result["frequency"] = float(f)
        result["disp_mm"] = float(
            peak_ms2 / (2 * np.pi * f) ** 2 * 1000
        )
    return result


# --------------------------------------------------
# Web interface and APIs
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
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
        elapsed = (
            time.monotonic() - stats["started"]
            if stats["started"] else 0
        )
        return jsonify({
            "samples": stats["samples"],
            "dropped": stats["dropped"],
            "measured_rate_hz": stats["samples"] / elapsed if elapsed else 0,
            "error": stats["error"],
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

        path = recording_path

    # Stream from disk instead of loading the whole file into memory.
    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=os.path.basename(path),
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
