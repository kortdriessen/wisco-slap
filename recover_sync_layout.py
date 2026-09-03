#!/usr/bin/env python3
"""CLI wrapper: recover a corrupt DataRec SYNC HDF5 by layout.

Implementation lives in :mod:`slap2_py.utils.sync_recovery`. Usage::

    python recover_sync_layout.py --good SYNC_00002.h5 --bad SYNC_00001.h5 \\
        --out recovered.h5
"""

from slap2_py.utils.sync_recovery import main

if __name__ == "__main__":
    main()
