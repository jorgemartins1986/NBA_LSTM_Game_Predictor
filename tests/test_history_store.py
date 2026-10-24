"""Exercise history persistence with real CSV files in temporary directories."""

import os
from pathlib import Path

import pytest

from src import history_store as store


@pytest.fixture
def saved_history(tmp_path):
    path = tmp_path / 'prediction_history.csv'
    # Match the original ten-column schema; include zeros and significant text.
    path.write_bytes(
        b'date,away_team,home_team,prediction,winner,confidence,home_win_prob,model_agreement,tier,correct\r\n'
        b'2026-10-16,Boston Celtics,Philadelphia 76ers,Boston Celtics,Boston Celtics,0.497,0.251,0.917,STRONG,1.0\r\n'
        b'2026-10-20,Philadelphia 76ers,New York Knicks,New York Knicks,Philadelphia 76ers,0.208,0.604,0.956,MODERATE,0.0\r\n'
        b'2026-10-21,"Team, A",NA,NA,,0.100,0.500,0.900,RISKY,\r\n')
    return store.read_history(path)


def test_schema_extension_preserves_every_existing_cell_and_exact_backup(saved_history):
    original = saved_history
    columns = original.columns + ['prediction_mode', 'recorded_at_utc']
    replay = dict(original.rows[-1], date='2026-10-22', prediction_mode='historical_replay')
    store.write_history(original, columns, original.rows + [replay])
    after = store.read_history(original.path)
    assert len(after.rows) == len(original.rows) + 1
    assert [{key: row[key] for key in original.columns} for row in after.rows[:-1]] == original.rows
    assert all(row['prediction_mode'] == '' for row in after.rows[:-1])
    assert after.rows[-1]['winner'] == after.rows[-1]['correct'] == ''
    assert original.path.with_suffix('.csv.bak').read_bytes() == original.content


def test_removing_only_added_metadata_preserves_all_ten_original_columns(saved_history):
    original = saved_history
    extended = original.columns + sorted(store.REMOVABLE_METADATA)
    store.write_history(original, extended, original.rows)
    with_metadata = store.read_history(original.path)
    store.write_history(with_metadata, original.columns, with_metadata.rows, drop_metadata=True)
    cleaned = store.read_history(original.path)
    assert cleaned.columns == original.columns
    assert cleaned.rows == original.rows
    assert original.path.with_suffix('.csv.bak').read_bytes() == with_metadata.content


def test_metadata_cleanup_cannot_drop_prediction_or_result_columns(saved_history):
    with pytest.raises(ValueError, match='Refusing'):
        store.write_history(saved_history, saved_history.columns[:-1], saved_history.rows, drop_metadata=True)
    assert saved_history.path.read_bytes() == saved_history.content


@pytest.mark.parametrize('damage', ['drop_row', 'clear_winner', 'clear_correct', 'change_prediction', 'drop_column', 'reorder_rows', 'invalid_permission'])
def test_refuses_to_lose_or_change_previous_results(saved_history, damage):
    rows = [dict(row) for row in saved_history.rows]
    columns = saved_history.columns.copy()
    updates = ()
    if damage == 'drop_row':
        rows.pop()
    elif damage == 'clear_winner':
        rows[0]['winner'] = ''
    elif damage == 'clear_correct':
        rows[1]['correct'] = ''
    elif damage == 'change_prediction':
        rows[0]['prediction'] = 'Different pick'
    elif damage == 'drop_column':
        columns.remove('winner')
    elif damage == 'reorder_rows':
        rows.reverse()
    else:
        updates = {(0, 'prediction')}
    with pytest.raises(ValueError, match='Refusing'):
        store.write_history(saved_history, columns, rows, result_updates=updates)
    assert saved_history.path.read_bytes() == saved_history.content
    assert not saved_history.path.with_suffix('.csv.bak').exists()


def test_stale_writer_cannot_overwrite_newer_history(saved_history):
    newer = saved_history.content + saved_history.content.splitlines(keepends=True)[1]
    saved_history.path.write_bytes(newer)
    with pytest.raises(RuntimeError, match='History changed'):
        store.write_history(saved_history, saved_history.columns, saved_history.rows)
    assert saved_history.path.read_bytes() == newer
    assert not saved_history.path.with_suffix('.csv.bak').exists()
    assert not saved_history.path.with_suffix('.csv.lock').exists()


@pytest.mark.parametrize('column,value', [('winner', ''), ('correct', ''), ('correct', 'nan')])
def test_result_updates_cannot_clear_or_invalidate_results(saved_history, column, value):
    rows = [dict(row) for row in saved_history.rows]
    rows[0][column] = value
    with pytest.raises(ValueError, match='Refusing'):
        store.write_history(saved_history, saved_history.columns, rows, result_updates={(0, column)})
    assert saved_history.path.read_bytes() == saved_history.content


def test_concurrent_writer_fails_without_touching_history(saved_history):
    lock = saved_history.path.with_suffix('.csv.lock')
    lock.write_bytes(b'other writer')
    with pytest.raises(FileExistsError):
        store.write_history(saved_history, saved_history.columns, saved_history.rows)
    assert saved_history.path.read_bytes() == saved_history.content
    assert lock.read_bytes() == b'other writer'


@pytest.mark.parametrize('failure', ['stage', 'backup', 'replace'])
def test_failed_write_keeps_original_byte_identical(saved_history, monkeypatch, failure):
    original_stage, original_replace = store._stage, store.os.replace

    def stage(path, content):
        if failure == 'stage':
            raise OSError('disk full')
        return original_stage(path, content)

    def replace(source, destination):
        if (failure == 'backup' and str(destination).endswith('.bak')) or (
                failure == 'replace' and Path(destination) == saved_history.path):
            raise PermissionError('file unavailable')
        return original_replace(source, destination)

    monkeypatch.setattr(store, '_stage', stage)
    monkeypatch.setattr(store.os, 'replace', replace)
    with pytest.raises(OSError):
        store.write_history(saved_history, saved_history.columns, saved_history.rows)
    assert saved_history.path.read_bytes() == saved_history.content
    assert not saved_history.path.with_suffix('.csv.lock').exists()
    assert not list(saved_history.path.parent.glob('*.tmp'))
    if failure == 'replace':
        assert saved_history.path.with_suffix('.csv.bak').read_bytes() == saved_history.content


def test_failed_flush_cleans_staged_file_without_touching_history(saved_history, monkeypatch):
    def fail_flush(_):
        raise OSError('disk full')
    monkeypatch.setattr(store.os, 'fsync', fail_flush)
    with pytest.raises(OSError, match='disk full'):
        store.write_history(saved_history, saved_history.columns, saved_history.rows)
    assert saved_history.path.read_bytes() == saved_history.content
    assert not list(saved_history.path.parent.glob('*.tmp'))
    assert not saved_history.path.with_suffix('.csv.lock').exists()


@pytest.mark.parametrize('content', [b'', b'date,date\n', b'date,winner\n2026-10-20\n'])
def test_malformed_history_is_rejected_without_rewriting(tmp_path, content):
    path = tmp_path / 'history.csv'
    path.write_bytes(content)
    with pytest.raises(ValueError, match='Invalid history'):
        store.read_history(path)
    assert path.read_bytes() == content


@pytest.mark.parametrize('winner,correct', [('NA', 1.0), ('Team, A', 0.0)])
def test_result_updater_changes_only_matching_result_cells(saved_history, monkeypatch, capsys, winner, correct):
    import pandas as pd
    from src import update_prediction_results as updater
    monkeypatch.setattr(updater, 'PREDICTION_HISTORY_FILE', str(saved_history.path))
    monkeypatch.setattr(updater, 'get_game_results', lambda _: {'Team, A vs NA': winner})
    original_stats = updater.show_stats
    def assert_numeric_summary(frame):
        for column in ('correct', 'confidence', 'home_win_prob', 'model_agreement'):
            assert pd.api.types.is_numeric_dtype(frame[column])
        assert frame['correct'].iloc[-1] == correct
        original_stats(frame)
    monkeypatch.setattr(updater, 'show_stats', assert_numeric_summary)
    updater.update_results(date_str='2026-10-21')
    after = store.read_history(saved_history.path)
    assert after.rows[:-1] == saved_history.rows[:-1]
    assert after.rows[-1] == dict(saved_history.rows[-1], winner=winner, correct=str(correct))
    assert saved_history.path.with_suffix('.csv.bak').read_bytes() == saved_history.content
    output = capsys.readouterr().out
    assert ('Overall: 2/3 correct (66.7%)' if correct else 'Overall: 1/3 correct (33.3%)') in output
    assert '40%+' in output and '10-20%' in output
    assert '90-95% agree' in output and '95%+ agree' in output


def test_no_results_does_not_rewrite_history(saved_history, monkeypatch, capsys):
    from src import update_prediction_results as updater
    monkeypatch.setattr(updater, 'PREDICTION_HISTORY_FILE', str(saved_history.path))
    monkeypatch.setattr(updater, 'get_game_results', lambda _: {})
    updater.update_results(date_str='2026-10-21')
    assert saved_history.path.read_bytes() == saved_history.content
    assert not saved_history.path.with_suffix('.csv.bak').exists()
    assert 'Overall: 1/2 correct (50.0%)' in capsys.readouterr().out


@pytest.mark.parametrize('text_cells', [False, True])
def test_stats_accepts_numeric_and_lossless_text_history_without_mutating_input(text_cells, capsys):
    import pandas as pd
    from src.update_prediction_results import show_stats

    frame = pd.DataFrame({
        'date': ['2026-10-20'] * 6,
        'correct': [1.0, 0.0, 1, 0, '', None],
        'confidence': [.4, .3, .2, .1, .5, .6],
        'model_agreement': [.96, .92, .88, .97, .99, .99],
        'tier': ['STRONG', 'GOOD', 'MODERATE', 'RISKY', 'EXCELLENT', 'EXCELLENT'],
    })
    if text_cells:
        frame.loc[:3, 'correct'] = ['1.0', '0.0', '1', '0']
        for column in ('confidence', 'model_agreement'):
            frame[column] = frame[column].astype(str)
    original = frame.copy(deep=True)
    show_stats(frame)
    assert 'Overall: 2/4 correct (50.0%)' in capsys.readouterr().out
    pd.testing.assert_frame_equal(frame, original)


def test_test_suite_cannot_write_real_history():
    from src.paths import PREDICTION_HISTORY_FILE
    with pytest.raises(PermissionError, match='temporary prediction history'):
        Path(PREDICTION_HISTORY_FILE).write_text('must not reach the real file')
    with pytest.raises(PermissionError, match='temporary prediction history'):
        os.open(PREDICTION_HISTORY_FILE, os.O_WRONLY | os.O_TRUNC)
