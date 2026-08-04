import os

# NLTK 2026 blocks `regex` import when it resolves under CWD (e.g. a project .venv).
# We don't rely on that CWE-427 mitigation here; opt out before llama_index pulls nltk.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ragwatcher")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
