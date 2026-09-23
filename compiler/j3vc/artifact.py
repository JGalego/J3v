"""Python writer for the `.j3a` container (mirror of crates/j3v-core/src/artifact.rs)."""
import json
import struct

import numpy as np

ALIGN = 64


def write(path, kind, meta, tensors):
    """tensors: list of (name, np.ndarray[float32 | int8])."""
    data, index = bytearray(), []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr)
        dt = {np.dtype(np.float32): "f32", np.dtype(np.int8): "i8"}[arr.dtype]
        data += b"\0" * (-len(data) % ALIGN)
        index.append({"name": name, "dtype": dt, "shape": list(arr.shape), "offset": len(data)})
        data += arr.astype("<f4" if dt == "f32" else "i1").tobytes()
    h = json.dumps({"kind": kind, "meta": meta, "tensors": index}).encode()
    out = bytearray(b"J3VA" + struct.pack("<II", 1, len(h)) + h)
    out += b"\0" * (-len(out) % ALIGN)
    with open(path, "wb") as f:
        f.write(bytes(out + data))
