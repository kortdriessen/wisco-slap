import os

import wisco_slap.defs as DEFS


def check_dir(dir_path: str) -> None:
    """Check if a directory exists, and create it if it doesn't.

    Args:
        dir_path (str): Path to the directory to check/create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return
