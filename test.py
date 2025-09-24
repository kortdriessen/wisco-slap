import importlib
import importlib.metadata as im
import importlib.util
import json
import os
import sys

dist_name = "slap2-py"  # distribution name (hyphen)
mod_name = "slap2_py"  # top-level import (underscore)
print("Interpreter:", sys.executable)

spec = importlib.util.find_spec(mod_name)
print("Found module?:", bool(spec))
if spec:
    m = importlib.import_module(mod_name)
    print("__file__:", getattr(m, "__file__", None))

try:
    dist = im.distribution(dist_name)
    du = dist.locate_file("direct_url.json")
    print("direct_url.json:", du)
    if os.path.exists(du):
        print("direct_url.json contents:", json.load(open(du)))
    else:
        print("No direct_url.json (not a direct/path install)")
except Exception as e:
    print("No dist metadata for", dist_name, "-", e)
