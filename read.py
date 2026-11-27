import numpy as np
from pathlib import Path

dev = Path("/sys/bus/iio/devices/iio:device0")
scale = float((dev / "in_accel_scale").read_text())   # m/s² per LSB

dt = np.dtype([("xyz", ">i4", 3), ("pad", "V4"), ("ts", "<i8")])  # 24 bytes
assert dt.itemsize == 24

def decode(buf):
    a = np.frombuffer(buf, dtype=dt)
    raw = a["xyz"] >> 4                  # arithmetic shift keeps the sign
    raw = raw & 0xFFFFF                  # keep the 20 valid bits
    raw = (raw ^ 0x80000) - 0x80000      # sign-extend from bit 19
    xyz = raw * scale
    t_ns = a["ts"]                               # int64 nanoseconds
    t_dt = t_ns.astype("datetime64[ns]")         # numpy datetimes, UTC
    return xyz, t_dt

with open("/dev/iio:device0", "rb", buffering=0) as f:
    while True:
        buf = f.read(dt.itemsize * 100)    # ~100 ms of data per read
        buf = buf[: len(buf) // dt.itemsize * dt.itemsize]
        xyz, t = decode(buf)
        # process or write to disk here