"""Read the current checkpoint format and verify its encoding artifacts."""
from pathlib import Path
from tabulargen.artifacts import require_files, verify_encoding


def load_checkpoint(path):
    import torch

    path = Path(path).resolve()
    require_files(path.parent, [path.name], 'Checkpoint')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 2:
        raise ValueError('Expected checkpoint format_version = 2')
    encoded = (path.parent / checkpoint['encoding']['path']).resolve()
    verify_encoding(encoded, checkpoint['encoding']['sha256'])
    return checkpoint, encoded
