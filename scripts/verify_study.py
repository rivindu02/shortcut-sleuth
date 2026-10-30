"""Verify frozen-study artifacts against every saved paired prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from shortcut_sleuth.data import digest, load_manifest, save_json
from shortcut_sleuth.metrics import aggregate_seeds, paired_metrics
from shortcut_sleuth.train import implementation_hash
from shortcut_sleuth.inference import Predictor
import torch
torch.set_num_threads(2)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='configs/frozen-v1.json')
parser.add_argument('--data', required=True)
parser.add_argument('--runs', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
manifest, objects = load_manifest(Path(args.data) / 'test-paired.json')
report = json.loads((Path(args.report) / 'metrics.json').read_text())
expected = {(a,c,s) for a in config['architectures'] for c in config['conditions'] for s in config['seeds']}
identities, commits, environments, verified = set(), set(), set(), []
first_ids = None
cpu_checks = []
fixed_indices = sorted({i for i in [0, 1, 2, 57, 58, 59, len(objects)-3, len(objects)-2, len(objects)-1] if 0 <= i < len(objects)})
for path in sorted(Path(args.runs).glob('*/evaluation.json')):
    row = json.loads(path.read_text())
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    curves = json.loads((path.parent / 'curves.json').read_text())
    identity = (row['architecture'], row['condition'], row['seed'])
    assert identity not in identities
    identities.add(identity)
    assert len(curves) == config['epochs'] and curves[-1]['epoch'] == config['epochs']
    resolved = {**config, 'architecture':identity[0], 'condition':identity[1], 'seed':identity[2]}
    assert metadata['config_sha256'] == digest(resolved) == digest(metadata['config'])
    assert metadata['checkpoint_selection'] == 'final epoch'
    assert metadata['implementation_sha256'] == implementation_hash() == config['freeze']['implementation_sha256']
    assert not metadata['code']['dirty']
    commits.add(metadata['code']['commit'])
    environments.add(json.dumps(metadata['environment'], sort_keys=True))
    assert row['stage'] == 'study' and row['protocol'] == config['protocol']
    assert row['test_manifest_sha256'] == manifest['sha256']
    prediction_file = path.parent / 'predictions.jsonl'
    checkpoint = path.parent / 'final.pt'
    assert hashlib.sha256(prediction_file.read_bytes()).hexdigest() == row['predictions_sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
    records = [json.loads(line) for line in prediction_file.read_text().splitlines()]
    ids = [r['object_id'] for r in records]
    assert ids == [o.object_id for o in objects] and len(set(ids)) == len(objects)
    assert [r['label'] for r in records] == [o.label for o in objects]
    if first_ids is None:
        first_ids = ids
    assert ids == first_ids
    probabilities = np.asarray([r['probabilities_by_background'] for r in records])
    labels = np.asarray([r['label'] for r in records])
    assert probabilities.shape == (len(objects), 3, 3)
    assert paired_metrics(probabilities, labels) == row['metrics']
    cpu = Predictor.load(checkpoint, device='cpu').objects([objects[i] for i in fixed_indices])
    saved = probabilities[fixed_indices]
    cpu_checks.append({'architecture':identity[0], 'condition':identity[1], 'seed':identity[2],
                       'max_probability_delta':float(np.abs(cpu-saved).max()),
                       'predicted_labels_equal':bool(np.array_equal(cpu.argmax(-1),saved.argmax(-1)))})
    assert metadata['parameters'] == row['parameters']
    assert row['runtime_seconds'] > 0
    if metadata['environment']['device'].startswith('cuda'):
        assert row['peak_gpu_memory_bytes'] > 0
    else:
        assert row['peak_gpu_memory_bytes'] is None
    verified.append({'architecture':identity[0], 'condition':identity[1], 'seed':identity[2],
                     'checkpoint_sha256':row['checkpoint_sha256'], 'predictions_sha256':row['predictions_sha256'],
                     'final_validation_accuracy':metadata['final_validation_accuracy']})
assert identities == expected and len(commits) == 1 and len(environments) == 1
assert report['full_matrix_complete'] and len(report['runs']) == len(expected)
for a,c in sorted({(a,c) for a,c,_ in expected}):
    selected = sorted([r for r in report['runs'] if r['architecture']==a and r['condition']==c], key=lambda r:r['seed'])
    group = report['groups'][f'{a}/{c}']
    assert group['seeds'] == sorted(config['seeds'])
    expected_group = aggregate_seeds([r['metrics'] for r in selected])
    assert group == {'seeds':sorted(config['seeds']), **expected_group}
labels_agree = all(row['predicted_labels_equal'] for row in cpu_checks)
save_json(args.output, {
    'passed':labels_agree, 'runs_verified':len(expected), 'objects_per_run':len(objects), 'paired_renders_per_object':3,
    'prediction_vectors_verified':len(expected)*len(objects)*3, 'frozen_config_sha256':digest(config),
    'implementation_sha256':implementation_hash(), 'test_manifest_sha256':manifest['sha256'],
    'training_commit':next(iter(commits)), 'all_training_trees_clean':True,
    'runs':verified, 'cpu_reinference_checks':cpu_checks, 'cpu_label_agreement':labels_agree,
    'inference_check_object_ids':[objects[i].object_id for i in fixed_indices],
    'probabilities_are_bitwise_identical':all(r['max_probability_delta'] == 0 for r in cpu_checks),
    'reinference_note':'CPU inference may use different device/batch kernels; primary metrics use the saved evaluation predictions',
})
print(f'Verified {len(expected)} runs and {len(expected)*len(objects)*3:,} prediction vectors; CPU labels agree: {labels_agree}')
raise SystemExit(0 if labels_agree else 1)
