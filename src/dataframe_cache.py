"""Versioned, disposable caches for engineered training data."""

import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd


_MAGIC = b"NBA_DATAFRAME_CACHE\n"
_FORMAT_VERSION = 1


def _metadata(data_key, window_size):
    return {
        "format_version": _FORMAT_VERSION,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "data_key": data_key,
        "window_size": window_size,
    }


def load_dataframe_cache(path, data_key, window_size):
    """Return a compatible cache or None so the caller rebuilds from raw games.

    The plain header is checked before unpickling any pandas objects. Legacy,
    incompatible, and interrupted writes are disposable, never training inputs.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as stream:
            if stream.read(len(_MAGIC)) != _MAGIC:
                print(f"Rebuilding {os.path.basename(path)}: legacy cache format.")
                return None
            header = json.loads(stream.readline(4096))
            if header != _metadata(data_key, window_size):
                print(f"Rebuilding {os.path.basename(path)}: library versions or feature settings changed.")
                return None
            cached = pickle.load(stream)
        frame = cached[data_key]
        if not isinstance(frame, pd.DataFrame) or "GAME_DATE" not in frame:
            raise ValueError("Invalid cached dataframe")
        if not pd.api.types.is_datetime64_any_dtype(frame["GAME_DATE"]):
            raise ValueError("Invalid cached dates")
        if pd.isna(cached["max_date"]) or frame["GAME_DATE"].max() != cached["max_date"]:
            raise ValueError("Invalid cache cutoff")
        return cached
    except Exception as error:
        print(f"Rebuilding {os.path.basename(path)}: unreadable cache ({type(error).__name__}).")
        return None


def save_dataframe_cache(path, data_key, frame, max_date, window_size):
    """Replace a cache atomically, leaving the previous file intact on failure."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=os.path.dirname(os.path.abspath(path)),
                                         prefix=os.path.basename(path) + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary_path = stream.name
            stream.write(_MAGIC)
            stream.write(json.dumps(_metadata(data_key, window_size)).encode("utf-8") + b"\n")
            pickle.dump({data_key: frame, "max_date": max_date}, stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
