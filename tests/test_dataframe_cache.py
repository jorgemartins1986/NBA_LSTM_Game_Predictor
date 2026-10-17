"""Cache compatibility and recovery must never change training data."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.dataframe_cache import load_dataframe_cache, save_dataframe_cache


@pytest.fixture
def frame():
    return pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2026-10-01", "2026-10-02"]),
        "GAME_ID": ["0012600001", "0012600002"],
        "PTS_ROLL_MEAN": [float("nan"), 99.12345678901234],
    })


def test_cache_preserves_dates_ids_and_feature_precision(tmp_path, frame):
    path = tmp_path / "features.pkl"
    save_dataframe_cache(path, "features", frame, frame.GAME_DATE.max(), 20)
    cached = load_dataframe_cache(path, "features", 20)
    pd.testing.assert_frame_equal(cached["features"], frame)
    assert cached["max_date"] == frame.GAME_DATE.max()


@pytest.mark.parametrize("change", ["window", "pandas", "numpy", "key"])
def test_changed_settings_rejected_before_unpickling(tmp_path, frame, monkeypatch, change):
    import src.dataframe_cache as cache

    path = tmp_path / "features.pkl"
    save_dataframe_cache(path, "features", frame, frame.GAME_DATE.max(), 20)
    window, key = 20, "features"
    if change == "window":
        window = 10
    elif change == "key":
        key = "matchups"
    else:
        monkeypatch.setattr(getattr(cache, "pd" if change == "pandas" else "np"),
                            "__version__", "incompatible")
    with patch.object(cache.pickle, "load") as unpickle:
        assert load_dataframe_cache(path, key, window) is None
    unpickle.assert_not_called()


@pytest.mark.parametrize("contents", [b"old pickle", b"NBA_DATAFRAME_CACHE\n{broken"])
def test_unreadable_cache_rebuilds_with_short_message(tmp_path, capsys, contents):
    path = tmp_path / "features.pkl"
    path.write_bytes(contents)
    assert load_dataframe_cache(path, "features", 20) is None
    message = capsys.readouterr().out
    assert "Rebuilding features.pkl:" in message
    assert len(message.splitlines()) == 1


def test_truncated_payload_rebuilds(tmp_path, frame):
    path = tmp_path / "features.pkl"
    save_dataframe_cache(path, "features", frame, frame.GAME_DATE.max(), 20)
    path.write_bytes(path.read_bytes()[:-30])
    assert load_dataframe_cache(path, "features", 20) is None


def test_failed_save_keeps_previous_cache(tmp_path, frame):
    path = tmp_path / "features.pkl"
    save_dataframe_cache(path, "features", frame, frame.GAME_DATE.max(), 20)
    original = path.read_bytes()
    with patch("src.dataframe_cache.pickle.dump", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            save_dataframe_cache(path, "features", frame, frame.GAME_DATE.max(), 20)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_wrong_cutoff_rebuilds(tmp_path, frame):
    path = tmp_path / "features.pkl"
    save_dataframe_cache(path, "features", frame, pd.Timestamp("2026-10-03"), 20)
    assert load_dataframe_cache(path, "features", 20) is None
