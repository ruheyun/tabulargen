import argparse
from tabulargen.config import resolve_config
from tabulargen.layout import RunPaths
from tabulargen.training.privacy import mechanism
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Print the noise multiplier; do not modify configuration')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = resolve_config(args.config)
    mechanism(exp_path=str(RunPaths(Path(c['experiment']['path'])).encoded),
              epochs=c['train']['epochs'], batch_size=c['train']['batch_size'],
              target_epsilon=c['privacy']['epsilon'], target_delta=c['privacy']['delta'])


if __name__ == "__main__":
    main()
