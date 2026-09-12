"""Behavioral tests for installed-package execution and stage boundaries."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch
import tomli_w

from tabulargen.config import resolve_config
from tabulargen.models import MLPDiffusion
from tabulargen.layout import RunPaths
from tabulargen.data.preprocess import data_process
from tabulargen.data.encoding import load_wrapper


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def run_stage(config, *flags):
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)
    return subprocess.run(
        [sys.executable, '-B', '-m', 'tabulargen', '--config', str(config), *flags],
        cwd=config.parent, env=env, text=True, capture_output=True, timeout=180,
    )


@pytest.fixture
def experiment(tmp_path):
    data = tmp_path / 'data' / 'adult'
    data.mkdir(parents=True)
    for split, size in [('train', 128), ('val', 32), ('test', 32)]:
        pd.DataFrame({'number': [float(i % 7) for i in range(size)],
                      'category': ['abcd'[i % 4] for i in range(size)],
                      'target': [i % 2 for i in range(size)]}).to_csv(data / f'adult_{split}.csv', index=False)
    (data / 'info.json').write_text(json.dumps({'task_type': 'binclass', 'n_num_features': 1,
                                               'n_cat_features': 1, 'n_classes': 2}))
    config = {
        'schema_version': 2, 'seed': 0, 'device': 'cpu',
        'data': {'path': 'data/adult'}, 'experiment': {'path': 'run'},
        'model': {'num_classes': 2, 'is_y_cond': True,
                  'rtdl_params': {'d_layers': [8, 8], 'dropout': 0.0, 'activation': 'SiLU'}},
        'diffusion': {'num_timesteps': 10, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'},
        'train': {'epochs': 1, 'batch_size': 32, 'num_workers': 0},
        'sample': {'num_samples': 128, 'batch_size': 32},
        'evaluation': {'model': 'simple', 'mode': 'synthetic'},
    }
    path = tmp_path / 'config.toml'
    path.write_text(tomli_w.dumps(config))
    return path, config


def assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_does_not_modify_nested_configuration():
    params = dict(d_in=3, num_classes=2, is_y_cond=True,
                  rtdl_params=dict(d_layers=[8, 8], dropout=0.0, activation='SiLU'))
    before = deepcopy(params)
    MLPDiffusion(**params)
    assert params == before


def test_resolution_and_reproducible_encoding(experiment):
    path, raw = experiment
    raw['seed'] = 9
    raw['sample']['seed'] = 3
    path.write_text(tomli_w.dumps(raw))
    c = resolve_config(path)
    assert c['train']['seed'] == c['encoding']['seed'] == c['evaluation']['seed'] == 9
    assert c['sample']['seed'] == 3
    assert c['data']['path'] == str(path.parent / 'data/adult')
    dirs = [path.parent / name for name in ['encoded_a', 'encoded_b']]
    for directory in dirs:
        data_process(c['data']['path'], str(directory), **c['encoding'])
    assert hashes(dirs[0]) == hashes(dirs[1])
    info = json.loads((dirs[0] / 'info.json').read_text())
    assert info['raw_feature_count'] == 2
    assert info['encoded_dim'] == 3
    assert 'n_features' not in info
    assert load_wrapper(dirs[0] / 'data_wrapper.pkl').seed == 9


def test_independent_stages_and_seed_isolation(experiment):
    path, config = experiment
    paths = RunPaths(path.parent / 'run')
    missing = run_stage(path, '--train')
    assert missing.returncode != 0 and '--encode' in missing.stderr
    for stage in ['--encode', '--train']:
        assert_success(run_stage(path, stage))
    before = {name: hashes(getattr(paths, name)) for name in ['encoded', 'checkpoints', 'logs']}
    checkpoint = torch.load(paths.checkpoints / 'checkpoint.pt', weights_only=True)
    assert checkpoint['model_params']['d_in'] == 3
    assert checkpoint['encoding']['path'] == '../encoded'
    assert 'd_in' not in checkpoint['model_params']['rtdl_params']
    training_config = (paths.root / 'config.json').read_bytes()
    for key in ['model', 'diffusion', 'train']:
        config.pop(key)
    path.write_text(tomli_w.dumps(config))
    for stage in ['--sample', '--eval']:
        assert_success(run_stage(path, stage))
        assert {name: hashes(getattr(paths, name)) for name in before} == before
        assert (paths.root / 'config.json').read_bytes() == training_config
    seed0 = hashes(paths.samples(0))
    simple_path = paths.evaluation('synthetic', 0, 'simple', 0) / 'results.json'
    report = json.loads(simple_path.read_text())
    assert set(report['metrics']) == {'train', 'val', 'test'}
    assert set(report['per_model']) == {'tree', 'rf', 'lr', 'mlp'}
    assert_success(run_stage(path, '--sample', '--sample-seed', '1'))
    assert hashes(paths.samples(0)) == seed0
    assert paths.samples(1).is_dir()
    assert_success(run_stage(path, '--eval', '--eval-seed', '1'))
    assert paths.evaluation('synthetic', 0, 'simple', 1).exists()
    # Both evaluators use the same output envelope and preserve each other.
    config['evaluation'] = {'model': 'catboost', 'mode': 'synthetic',
                            'params': {'iterations': 5, 'depth': 2, 'thread_count': 1}}
    path.write_text(tomli_w.dumps(config))
    assert_success(run_stage(path, '--eval'))
    cat = json.loads((paths.evaluation('synthetic', 0, 'catboost', 0) / 'results.json').read_text())
    assert set(cat) == set(report)
    assert set(cat['metrics']) == set(report['metrics'])
    assert simple_path.exists()
    config['evaluation'] = {'model': 'all', 'mode': 'synthetic',
                            'params': {'catboost': {'iterations': 5, 'depth': 2, 'thread_count': 1}}}
    path.write_text(tomli_w.dumps(config))
    combined_run = run_stage(path, '--eval')
    assert_success(combined_run)
    assert 'Equal-weight average across 5 models' in combined_run.stdout
    combined_path = paths.evaluation('synthetic', 0, 'all', 0)
    combined = json.loads((combined_path / 'results.json').read_text())
    assert combined['per_model'] == cat['per_model'] | report['per_model']
    for split in ['train', 'val', 'test']:
        for metric in ['f1', 'accuracy', 'roc_auc']:
            expected = sum(row[split][metric] for row in combined['per_model'].values()) / 5
            assert combined['metrics'][split][metric] == pytest.approx(expected)
    saved_params = json.loads((combined_path / 'config.json').read_text())['params']
    assert saved_params['catboost']['iterations'] == 5
    assert hashes(paths.samples(0)) == seed0
    sample_file = paths.samples(0) / 'reverse.csv'
    saved_sample = sample_file.read_bytes()
    sample_file.write_bytes(saved_sample + b'\n')
    stale = run_stage(path, '--eval', '--eval-seed', '2')
    assert stale.returncode != 0 and 'Sample data changed' in stale.stderr
    sample_file.write_bytes(saved_sample)
    refused = run_stage(path, '--encode')
    assert refused.returncode != 0 and 'new experiment.path' in refused.stderr
    repeat = run_stage(path, '--sample')
    assert repeat.returncode != 0 and 'sample seed already exists' in repeat.stderr
    with (paths.encoded / 'info.json').open('a') as stream:
        stream.write('\n')
    mismatch = run_stage(path, '--sample', '--sample-seed', '2')
    assert mismatch.returncode != 0 and 'do not match this checkpoint' in mismatch.stderr
    assert hashes(paths.samples(0)) == seed0


def test_imports_have_no_side_effects(tmp_path):
    result = subprocess.run([sys.executable, '-B', '-c',
        'import tabulargen; import tabulargen.tools.tune_ddpm; import tabulargen.tools.tune_catboost; '
        'import tabulargen.tools.num2cat; import tabulargen.tools.merge_csv'], cwd=tmp_path,
        capture_output=True, text=True, timeout=30)
    assert_success(result)
    assert list(tmp_path.iterdir()) == []


def test_real_catboost_needs_no_generation(experiment):
    path, config = experiment
    config['evaluation'] = {'model': 'catboost', 'mode': 'real',
                            'params': {'iterations': 5, 'depth': 2, 'thread_count': 1}}
    path.write_text(tomli_w.dumps(config))
    assert_success(run_stage(path, '--eval'))
    paths = RunPaths(path.parent / 'run')
    assert not paths.encoded.exists()
    assert not paths.checkpoints.exists()
    report = json.loads((paths.evaluation('real', 0, 'catboost', 0) / 'results.json').read_text())
    assert report['sample_seed'] is None


@pytest.mark.parametrize('script', ['tune_ddpm', 'tune_catboost'])
def test_tuning_reads_new_result_layout(experiment, script):
    path, config = experiment
    config['evaluation'] = {'model': 'catboost', 'mode': 'synthetic',
                            'params': {'iterations': 5, 'depth': 2, 'thread_count': 1}}
    path.write_text(tomli_w.dumps(config))
    output = path.parent / script
    command = [sys.executable, '-B', '-m', f'tabulargen.tools.{script}',
               '--config', str(path), '--output', str(output), '--num-trials', '1']
    if script == 'tune_ddpm':
        command += ['--sample-seeds', '0']
    result = subprocess.run(command, cwd=path.parent, text=True, capture_output=True, timeout=180,
                            env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    assert_success(result)
    assert (output / 'best.json').exists()
    assert list((output / 'trial_0' / 'evaluation').rglob('results.json'))
    assert path.read_text() == tomli_w.dumps(config)


@pytest.mark.parametrize('contents', [{'format_version': 1}, {'weight': torch.zeros(1)}])
def test_rejects_obsolete_checkpoint_formats(tmp_path, contents):
    from tabulargen.checkpoints import load_checkpoint
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(contents, checkpoint)
    with pytest.raises(ValueError, match='format_version = 2'):
        load_checkpoint(checkpoint)


def test_average_gives_each_classifier_equal_weight():
    from tabulargen.evaluation.metrics import average_metrics
    results = {
        model: {split: {metric: float(model == 'catboost') for metric in ['f1', 'accuracy', 'roc_auc']}
                for split in ['train', 'val', 'test']}
        for model in ['catboost', 'tree', 'rf', 'lr', 'mlp']
    }
    for split in average_metrics(results).values():
        assert all(value == pytest.approx(0.2) for value in split.values())
