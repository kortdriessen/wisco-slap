# launch_selector_webagg.py
from __future__ import annotations

import matplotlib

matplotlib.use("WebAgg")  # must be set before importing pyplot

from matplotlib import rcParams

rcParams["webagg.address"] = "127.0.0.1"
rcParams["webagg.port"] = 8988
rcParams["webagg.port_retries"] = 50

import matplotlib.pyplot as plt
from source_selector_gui import SourceSelector


def launch_webagg(image, masks, output_path="selected_source_ids.txt", port=8988):
    rcParams["webagg.port"] = int(port)
    sel = SourceSelector(image=image, masks=masks, output_path=output_path)
    print(
        f"[launch_selector_webagg] Listening on http://127.0.0.1:{port}/ "
        f"(forward this port and open http://localhost:{port}/)"
    )
    plt.show()
    return sel


if __name__ == "__main__":
    import numpy as np

    H, W = 300, 600
    img = np.full((H, W), np.nan, dtype=float)
    img[80:140, 100:450] = 0.5
    N = 5
    masks = np.zeros((N, H, W), dtype=bool)
    masks[0, 100:120, 200:220] = True
    launch_webagg(img, masks, port=8988)
