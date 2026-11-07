"""NanoXL host driver (docs/driver.md): ``NanoXL`` over a ``SimBackend`` (funcsim + CSR emulation)
or a ``PynqBackend`` (board; untested until hardware exists)."""
from . import csr
from .backends import Backend, PynqBackend, SimBackend
from .host import SHAKESPEARE_CHARS, CharCodec, DriverError, DriverTimeout, NanoXL, sample

__all__ = ["csr", "Backend", "PynqBackend", "SimBackend", "NanoXL", "CharCodec", "SHAKESPEARE_CHARS",
           "DriverError", "DriverTimeout", "sample", "open_bundle"]


def open_bundle(path, backend=None, hw=None, **kw):
    """Load an SG1 bundle (.npz) and return a ready ``NanoXL`` (default: SimBackend on KV260)."""
    from pathlib import Path

    from ..export import load_bundle
    from ..hwconfig import KV260
    hw = hw or KV260
    arrays, meta = load_bundle(path)
    backend = backend or SimBackend(hw)
    return NanoXL(backend, arrays, meta, hw=hw, name=Path(path).stem, **kw)
