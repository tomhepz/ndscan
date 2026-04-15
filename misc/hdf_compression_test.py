import io
import gzip
import h5py
import numpy as np
from pathlib import Path

FILE = "/home/lab/artiq-files/dnamic-lab/results/2026-04-15/12/000002761-ThreeImageRearrangementDashboardFragment.h5"

def gzip_size(data: bytes, level: int = 4) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as f:
        f.write(data)
    return len(buf.getvalue())

def sample_array(arr, max_bytes=8_000_000):
    """Take a strided sample so we don't load huge datasets fully."""
    if arr.size == 0:
        return arr
    itemsize = arr.dtype.itemsize
    target_items = max(1, max_bytes // max(1, itemsize))
    if arr.size <= target_items:
        return arr
    step = max(1, arr.size // target_items)
    flat = arr.reshape(-1)
    return flat[::step]

def inspect_dataset(name, ds):
    storage = ds.id.get_storage_size()
    logical = ds.size * ds.dtype.itemsize if ds.dtype.kind != "O" else None

    print(f"\nDataset: {name}")
    print(f"  shape={ds.shape}, dtype={ds.dtype}")
    print(f"  chunks={ds.chunks}, compression={ds.compression}, compression_opts={ds.compression_opts}")
    print(f"  on_disk={storage/1024/1024:.2f} MB", end="")
    if logical is not None:
        print(f", logical={logical/1024/1024:.2f} MB", end="")
        if storage > 0:
            print(f", disk/logical={storage/logical:.3f}", end="")
    print()

    # Skip unsupported / awkward types
    if ds.dtype.kind == "O":
        print("  sample_estimate=skipped (object / variable-length type)")
        return

    try:
        arr = ds[...]
    except Exception as e:
        print(f"  sample_estimate=skipped (read failed: {e})")
        return

    s = sample_array(np.asarray(arr))
    raw = np.ascontiguousarray(s).tobytes()
    if not raw:
        print("  sample_estimate=empty")
        return

    gz = gzip_size(raw, level=4)
    ratio = gz / len(raw)
    print(f"  sample_gzip_ratio≈{ratio:.3f}  ({len(raw)/1024/1024:.2f} MB sample)")

def main():
    path = Path(FILE)
    print(f"Inspecting: {path} ({path.stat().st_size/1024/1024:.2f} MB)")

    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                inspect_dataset(name, obj)
        f.visititems(visitor)

if __name__ == "__main__":
    main()