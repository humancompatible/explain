# humancompatible/explain/fcx/scripts/blackbox_model_train.py

import importlib.machinery
import importlib.util
import os

# load the real file (with the dash) under a legal module name
_path = os.path.join(os.path.dirname(__file__), 'blackbox-model-train.py')
spec = importlib.util.spec_from_file_location(
    __name__,  # this module’s own name
    _path
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# re‑export everything except private names
__all__ = [n for n in dir(mod) if not n.startswith('_')]
for _name in __all__:
    globals()[_name] = getattr(mod, _name)
