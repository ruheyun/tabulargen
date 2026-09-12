"""File contracts shared by encoding, training, and sampling."""

import hashlib
from pathlib import Path


ENCODED_FILES = (
    'train.csv', 'val.csv', 'test.csv', 'info.json',
    'data_wrapper.pkl', 'label_wrapper.pkl',
)


def require_files(directory, names, stage):
    directory = Path(directory)
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'{stage}: missing files in {directory}: {", ".join(missing)}. '
            'Run the upstream stage first; no files were regenerated.'
        )


def encoded_fingerprints(directory):
    require_files(directory, ENCODED_FILES, 'Encoded data')
    return {
        name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        for name in ENCODED_FILES
    }


def verify_encoding(directory, expected):
    actual = encoded_fingerprints(directory)
    changed = [name for name in ENCODED_FILES if actual[name] != expected.get(name)]
    if changed:
        raise ValueError(
            'Encoded artifacts do not match this checkpoint: ' + ', '.join(changed)
        )
