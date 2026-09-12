"""One training run, with separate outputs for each sampling/evaluation seed."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def encoded(self):
        return self.root / 'encoded'

    @property
    def checkpoints(self):
        return self.root / 'checkpoints'

    @property
    def logs(self):
        return self.root / 'logs'

    def samples(self, seed):
        return self.root / 'samples' / f'seed_{seed}'

    def evaluation(self, mode, sample_seed, model, seed):
        source = f'seed_{sample_seed}' if mode == 'synthetic' else 'real'
        return self.root / 'evaluation' / source / model / f'seed_{seed}'
