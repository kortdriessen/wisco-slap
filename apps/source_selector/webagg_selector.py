# webagg_selector.py
from __future__ import annotations

import multiprocessing as mp

# IMPORTANT: set the backend *inside the subprocess*, not here.
# We'll keep this module backend-agnostic so importing it in Streamlit is safe.


def _run_webagg(image, masks, output_path, port):
    # This code runs in a brand-new interpreter process (main thread).
    import matplotlib

    matplotlib.use("WebAgg")
    from matplotlib import rcParams

    rcParams["webagg.address"] = "127.0.0.1"
    rcParams["webagg.port"] = int(port)
    rcParams["webagg.port_retries"] = 50

    import matplotlib.pyplot as plt
    from source_selector_gui import SourceSelector

    _ = SourceSelector(image=image, masks=masks, output_path=output_path)
    print(f"[webagg_selector subprocess] Serving at http://127.0.0.1:{port}/")
    # Blocking show is fine here because we're in the subprocess main thread.
    plt.show()


def start_selector_process(
    image, masks, output_path="selected_source_ids.txt", port=8988
):
    """
    Start the WebAgg selector in a separate process.

    Returns
    -------
    url : str
        http://127.0.0.1:<port>/  (forward this to your local machine)
    proc : multiprocessing.Process
        Handle to the running process. Keep this around so the process
        isn't garbage collected. You can terminate it when desired.
    """
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    proc = ctx.Process(
        target=_run_webagg, args=(image, masks, output_path, int(port)), daemon=True
    )
    proc.start()
    url = f"http://127.0.0.1:{int(port)}/"
    return url, proc


def stop_selector_process(proc):
    """Terminate a previously started selector process (if it is still alive)."""
    try:
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
    except Exception:
        pass
