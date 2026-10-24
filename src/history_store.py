"""Lossless CSV history snapshots and guarded, atomic persistence."""

import csv
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


# These were briefly added to the original ten-column history format.
REMOVABLE_METADATA = {'prediction_mode', 'recorded_at_utc', 'model_name',
                      'model_manifest_sha256', 'model_created_at_utc', 'model_training_cutoff'}


@dataclass
class HistorySnapshot:
    path: Path
    content: bytes | None
    columns: list[str]
    rows: list[dict[str, str]]


def _decode(content):
    records = list(csv.reader(io.StringIO(content.decode('utf-8-sig'), newline='')))
    if not records or not records[0] or len(set(records[0])) != len(records[0]):
        raise ValueError('Invalid history header; original file was not changed.')
    columns = records[0]
    # Do not silently drop or shift cells in an already damaged file.
    if any(len(row) != len(columns) for row in records[1:]):
        raise ValueError('Invalid history row width; original file was not changed.')
    return columns, [dict(zip(columns, row)) for row in records[1:]]


def read_history(path):
    """Read all cells as text, including literal NA and numeric formatting."""
    path = Path(path)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return HistorySnapshot(path, None, [], [])
    columns, rows = _decode(content)
    return HistorySnapshot(path, content, columns, rows)


def _stage(path, content):
    """Flush a complete replacement beside its destination before promotion."""
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def write_history(snapshot, columns, rows, *, result_updates=(), drop_metadata=False):
    """Preserve old cells; only explicitly listed winner/correct cells may change.

    Writes are serialized, stale snapshots are rejected, the previous bytes are
    backed up to <history>.bak, and the destination is replaced atomically.
    """
    columns = list(columns)
    rows = [{key: str(row.get(key, '')) for key in columns} for row in rows]
    allowed = set(result_updates)
    removed = set(snapshot.columns) - set(columns)
    if (len(set(columns)) != len(columns)
            or (removed and (not drop_metadata or not removed.issubset(REMOVABLE_METADATA)))
            or len(rows) < len(snapshot.rows)
            or any(col not in {'winner', 'correct'} for _, col in allowed)):
        raise ValueError('Refusing to remove or replace existing history.')
    for index, old in enumerate(snapshot.rows):
        for column in snapshot.columns:
            if column in removed:
                continue
            if (index, column) not in allowed and rows[index][column] != old[column]:
                raise ValueError(f'Refusing to change existing history row {index}, {column}.')
    for index, column in allowed:
        value = rows[index][column]
        if not value or (column == 'correct' and value not in {'0', '1', '0.0', '1.0'}):
            raise ValueError('Refusing to clear a result or save an invalid outcome.')

    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue().encode('utf-8')
    if _decode(content) != (columns, rows):
        raise ValueError('History serialization verification failed.')

    path = snapshot.path
    lock = path.with_name(path.name + '.lock')
    # Exclusive creation fails safely if another writer is active.
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    staged = backup_staged = None
    try:
        current = path.read_bytes() if path.exists() else None
        if current != snapshot.content:
            raise RuntimeError('History changed during this operation; rerun to avoid losing changes.')
        staged = _stage(path, content)
        if snapshot.content is not None:
            backup = path.with_name(path.name + '.bak')
            backup_staged = _stage(backup, snapshot.content)
            os.replace(backup_staged, backup)
        os.replace(staged, path)
    finally:
        for temporary in (staged, backup_staged):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        os.close(lock_fd)
        lock.unlink()
