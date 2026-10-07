#!/usr/bin/env python3
"""Prepare or explicitly run the pinned native CPU flatten recipe; no downloads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'reproduction' / 'flatten-source-lock.json'
ENV = {'LASAGNA_MAX_PRECISION_FLOAT': '32', 'LASAGNA_COMPILE_FLATTEN': '0',
       'LASAGNA_FUSED_FLATTEN_ADAM_CLAMP': '0', 'LASAGNA_FLATTEN_DIAGNOSTICS': '1',
       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_upstream(villa, lock):
    for entry in lock['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid relative path in source lock')
        path = villa / relative
        if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
                or sha(path) != entry['sha256']):
            raise ValueError(f'Pinned source mismatch: {relative}')


def source_record(source):
    files = {name: sha(source / name) for name in ('x.tif', 'y.tif', 'z.tif', 'meta.json')}
    meta = json.loads((source / 'meta.json').read_text())
    if not isinstance(meta, dict):
        raise ValueError('Expected an object in TIFXYZ meta.json')
    scale = meta.get('scale')
    if (not isinstance(scale, list) or len(scale) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x <= 0 for x in scale)
            or scale[0] != scale[1]):
        raise ValueError('Expected equal positive two-axis TIFXYZ scale')
    step = 1.0 / scale[0]
    if step not in (2.5, 5.0, 10.0):
        raise ValueError('This study recipe only qualifies source spacings 2.5, 5 or 10')
    return {'path': str(source), 'sha256': files, 'source_step': step}


def executable_path(value):
    """Resolve PATH names, but preserve a venv executable's symlink location."""
    value = str(Path(value).expanduser())
    found = shutil.which(value)
    if found is None:
        raise ValueError(f'Python executable not found or not executable: {value}')
    path = Path(os.path.abspath(found))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'Expected an executable file: {path}')
    return path


def canonical_sidecar(source):
    return {'external_surfaces': [{'path': str(source)}], 'voxel_size_um': 9.6}


def finite_origin(origin):
    return (isinstance(origin, (list, tuple)) and len(origin) == 3
            and all(not isinstance(x, bool) and isinstance(x, (int, float))
                    and math.isfinite(x) for x in origin))


def prepare(villa, source, run_dir, python, origin, lock_path=None):
    if os.path.lexists(run_dir):
        raise ValueError('Run directory already exists; choose a new directory')
    villa, source, run_dir = (Path(x).resolve() for x in (villa, source, run_dir))
    if run_dir == source or source in run_dir.parents:
        raise ValueError('Run directory must be outside the input surface directory')
    python = executable_path(python)
    if not finite_origin(origin):
        raise ValueError('A finite global L2 ZYX origin is required for provenance')
    if run_dir.exists():
        raise ValueError('Run directory already exists; choose a new directory')
    lock = json.loads(Path(LOCK if lock_path is None else lock_path).read_text())
    check_upstream(villa, lock)
    record = source_record(source)
    # No XYZ translation is made: the native consumer uses stored coordinates.
    run_dir.mkdir(parents=True, exist_ok=False)
    side = run_dir / 'input.json'
    write_json(side, canonical_sidecar(source))
    plan = {'schema': 'ink-surface-support/flatten-plan/1', 'villa': str(villa),
            'source': record, 'origin_global_l2_zyx': list(origin),
            'frame': 'stored local L2 XYZ; origin is evaluation provenance only',
            'voxel_size_um': 9.6, 'environment': ENV, 'source_lock': lock,
            'sidecar_sha256': sha(side), 'python': str(python), 'python_sha256': sha(python),
            'run_dir': str(run_dir), 'prepared_by_sha256': sha(__file__),
            'scope': 'Pinned CPU recipe, native output spacing from source scale; no quality claim'}
    write_json(run_dir / 'plan.json', plan)
    return run_dir / 'plan.json'


def parse_completion(text):
    stages = re.findall(r"\[optimizer\] stage(\d+) 'flatten' complete in ((?:\d+m)?\d+(?:\.\d+)?)s", text)
    declared = re.findall(r'\[optimizer\] stage(\d+): params=.*steps=(\d+)', text)
    iterations = re.findall(r'\bstage([012])\s+1000/1000(?:\s|$)', text)
    return {'final_iteration_stage_ids': iterations, 'final_iterations_reached': set(iterations) == {'0', '1', '2'},
            'completed_stage_ids': [s for s, _ in stages], 'display_times': [t for _, t in stages],
            'declared_steps': declared,
            'three_fixed_stages': [s for s, _ in stages] == ['0', '1', '2']
                and declared == [('0', '1000'), ('1', '1000'), ('2', '1000')]}


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != 'ink-surface-support/flatten-plan/1':
        raise ValueError('Unknown plan schema')
    root = plan_path.parent
    if Path(plan['run_dir']).resolve() != root:
        raise ValueError('Plan was moved: prepare a new plan instead')
    if plan['environment'] != ENV or plan['prepared_by_sha256'] != sha(__file__):
        raise ValueError('Recipe/launcher changed; prepare a new reviewed plan')
    # Use the repository lock as authority, not a modified plan's embedded lock.
    current_lock = json.loads(LOCK.read_text())
    if plan['source_lock'] != current_lock:
        raise ValueError('Source lock differs from this repository')
    villa, source = Path(plan['villa']), Path(plan['source']['path'])
    if not finite_origin(plan['origin_global_l2_zyx']) or plan['voxel_size_um'] != 9.6:
        raise ValueError('Invalid frame/voxel provenance in plan')
    if source.resolve() == root or source.resolve() in root.parents:
        raise ValueError('Run directory must be outside the input surface directory')
    python = executable_path(plan['python'])
    if str(python) != plan['python'] or sha(python) != plan['python_sha256']:
        raise ValueError('Python executable changed; prepare a new plan')
    check_upstream(villa, current_lock)
    side = root / 'input.json'
    if (source_record(source) != plan['source'] or sha(side) != plan['sidecar_sha256']
            or json.loads(side.read_text()) != canonical_sidecar(source)):
        raise ValueError('Input changed after plan creation or noncanonical sidecar recipe')
    dest = root / 'consumer'
    reserved = ('consumer', 'stdout.log', 'stderr.log', 'result.json', 'started.json')
    if any(os.path.lexists(root / name) for name in reserved):
        raise ValueError('Run already started or output path reserved; do not overwrite evidence')
    command = [str(python), str(villa / 'lasagna/fit.py'),
               str(villa / 'lasagna/configs/flatten_fast.json'), str(side),
               '--device', 'cpu', '--out-dir', str(dest)]
    environment = dict(os.environ, **ENV, PYTHONDONTWRITEBYTECODE='1')
    plan_sha = sha(plan_path)
    write_json(root / 'started.json', {'state': 'STARTED_NOT_COMPLETED', 'command': command,
        'plan_sha256': plan_sha, 'environment_set': ENV, 'start_new_session': False,
        'launcher_pgid': os.getpgrp() if hasattr(os, 'getpgrp') else None,
        'scope': 'Requested environment, not an independent measured dtype or resource attestation'})
    result = {'state': 'INCOMPLETE_OR_FAILED', 'cli_exit_code': None, 'command': command,
        'plan_sha256': plan_sha, 'environment_set': ENV,
        'not_claimed': ['reference quality', 'sheet identity', 'measured peak resources',
                        'repeatable timing', 'measured module dtype', 'sandbox isolation']}
    passed = False
    started = time.monotonic()
    try:
        with (root / 'stdout.log').open('x') as stdout, (root / 'stderr.log').open('x') as stderr:
            # No new group/local kill: the caller or scheduler owns cancellation.
            completed = subprocess.run(command, cwd=villa / 'lasagna', env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=False)
        result['cli_exit_code'] = completed.returncode
        text = (root / 'stdout.log').read_text(errors='replace')
        parsed = parse_completion(text)
        result['checks'] = parsed
        target = dest / 'tifxyz/flatten.tifxyz'
        output_files = ('x.tif', 'y.tif', 'z.tif', 'meta.json')
        present = (target.is_dir() and not target.is_symlink() and not dest.is_symlink()
                   and not (dest / 'tifxyz').is_symlink()
                   and all((target / n).is_file() and not (target / n).is_symlink() for n in output_files))
        checkpoint = dest / 'model_final.pt'
        checkpoint_present = checkpoint.is_file() and not checkpoint.is_symlink()
        outputs = {n: sha(target / n) for n in output_files} if present else {}
        output_scale_ok = (present and json.loads((target / 'meta.json').read_text()).get('scale')
                           == [1 / plan['source']['source_step']] * 2)
        unchanged = source_record(source) == plan['source']
        check_upstream(villa, current_lock)
        side_unchanged = sha(side) == plan['sidecar_sha256']
        plan_unchanged = sha(plan_path) == plan_sha
        emitted = re.findall(r'source_step=([0-9.eE+-]+) output_step=([0-9.eE+-]+)', text)
        steps_ok = bool(emitted) and all(float(a) == float(b) == plan['source']['source_step'] for a, b in emitted)
        passed = (completed.returncode == 0 and parsed['three_fixed_stages']
                  and parsed['final_iterations_reached'] and present and checkpoint_present
                  and output_scale_ok and unchanged and steps_ok and side_unchanged and plan_unchanged)
        result.update(native_steps_match=steps_ok, input_unchanged=unchanged,
                      sidecar_unchanged=side_unchanged, plan_unchanged=plan_unchanged,
                      output_scale_matches=output_scale_ok, output_sha256=outputs,
                      checkpoint_present=checkpoint_present,
                      checkpoint_sha256=sha(checkpoint) if checkpoint_present else None)
    except Exception as error:
        # Preserve a machine-readable failure even for spawn/post-check errors.
        result['error'] = f'{type(error).__name__}: {error}'
    result['wall_seconds_observed'] = time.monotonic() - started
    for name in ('stdout.log', 'stderr.log'):
        if (root / name).is_file():
            result[name + '_sha256'] = sha(root / name)
    result['state'] = 'CLI_COMPLETE_BASIC_OUTPUT_CHECKS' if passed else 'INCOMPLETE_OR_FAILED'
    write_json(root / 'result.json', result)
    return 0 if passed else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare', help='Validate inputs and write a plan; does not start compute')
    p.add_argument('--villa', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--origin-zyx', required=True, nargs=3, type=float)
    p.add_argument('--receipt', type=Path, help='Optional new JSON receipt path, also written on preparation failure')
    p = sub.add_parser('run', help='Explicitly execute the pinned CPU consumer; reserve resources first')
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--receipt', type=Path, help='Optional new JSON receipt path for run/preflight result')
    args = parser.parse_args(argv)
    if args.receipt and (os.path.lexists(args.receipt) or not args.receipt.parent.is_dir()):
        parser.error('--receipt needs an existing parent and a new filename; never overwritten')
    try:
        if args.action == 'prepare':
            path = prepare(args.villa, args.source, args.run_dir, args.python, args.origin_zyx)
            record = {'state': 'PREPARED_NO_COMPUTE', 'plan': str(path), 'plan_sha256': sha(path)}
            code = 0
        else:
            code = run(args.plan)
            record = {'state': 'RUN_CHECKS_PASSED' if code == 0 else 'INCOMPLETE_OR_FAILED',
                      'plan': str(args.plan), 'exit_code': code}
    except (ValueError, OSError, KeyError, TypeError) as error:
        record = {'state': 'PREPARE_REJECTED' if args.action == 'prepare' else 'RUN_REJECTED',
                  'error': f'{type(error).__name__}: {error}', 'action': args.action}
        code = 2
    if args.receipt:
        write_json(args.receipt, record)
    print(json.dumps(record, indent=2), file=sys.stdout if code == 0 else sys.stderr)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
