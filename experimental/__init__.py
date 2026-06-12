"""Experimental modules — not validated on real hardware, unstable APIs.

These modules are kept importable for development and validation work,
but ship with no guarantees: they may be removed, renamed, or rewritten
without notice. Set VRM_EXPERIMENTAL=1 to silence this warning once you
know what you're doing.

See experimental/README.md for the status of each module.
"""
import os
import warnings

if os.environ.get("VRM_EXPERIMENTAL") != "1":
    warnings.warn(
        "Importing an experimental VRAMancer module without VRM_EXPERIMENTAL=1. "
        "These modules are unvalidated on real hardware and have unstable APIs. "
        "See experimental/README.md.",
        stacklevel=2,
    )
