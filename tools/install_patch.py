#!/usr/bin/env python3
"""Install or restore a hash-verified final-mask patch in a Villa checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
CONTROL = '.ink-surface-support-install'
BASE_SHA256 = '8156f58f3a7cf1ff5301a211935e63280304328e729f4bbc6f4699dbb0c734e6'
PRODUCER = 'spiral-fitting/grow_track_graph.py'


class InstallError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise InstallError(message)


def real_directory(path):
    require(path.is_dir() and not path.is_symlink(), f'Expected a real directory: {path}')


def regular(path):
    require(path.is_file() and not path.is_symlink(), f'Expected a regular, non-symlink file: {path}')


def writable(path):
    require(bool(path.stat().st_mode & 0o222) and os.access(path, os.W_OK), f'Not writable: {path}')


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def package_files(output_spacing):
    manifest = json.loads((PACKAGE / 'vendor/SOURCE.json').read_text())
    require(manifest['base_sha256'] == BASE_SHA256, 'Unexpected manifest baseline')
    regular(PACKAGE / 'vendor/villa-pinned/grow_track_graph.py')
    require(digest(PACKAGE / 'vendor/villa-pinned/grow_track_graph.py') == BASE_SHA256,
            'Vendored original baseline hash mismatch')
    require(str(output_spacing) in manifest['configurations'], 'Unsupported output spacing')
    selected = manifest['configurations'][str(output_spacing)]
    files = []
    allowed = {PRODUCER, 'spiral-fitting/support_safe_finalize.py',
               'spiral-fitting/test_support_safe_finalize.py',
               'spiral-fitting/frozen_support_safe_finalize.py'}
    for entry in selected['files']:
        rel = PurePosixPath(entry['source'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.parts[:2] == ('vendor', f'output-spacing-{output_spacing}'),
                'Unsafe package source path')
        require(entry['target'] in allowed, 'Unsafe installation target')
        source = PACKAGE / rel
        regular(source)
        require(digest(source) == entry['sha256'], f'Package hash mismatch: {source}')
        files.append(dict(entry))
    require(len({f['target'] for f in files}) == len(files), 'Duplicate installation target')
    require(sum(f['target'] == PRODUCER for f in files) == 1, 'Missing unique producer')
    patch = PACKAGE / selected['patch']
    require(patch.parent == PACKAGE / 'patches' and not patch.is_symlink(), 'Unsafe patch path')
    require(digest(patch) == selected['patch_sha256'], 'Review diff hash mismatch')
    return manifest, files


def preflight(checkout, output_spacing):
    checkout = Path(checkout).absolute()
    real_directory(checkout)
    folder = checkout / 'spiral-fitting'
    real_directory(folder)
    writable(checkout)
    writable(folder)
    require(not os.path.lexists(checkout / CONTROL),
            f'Installation/recovery record already exists. Use --restore first: {checkout / CONTROL}')
    manifest, files = package_files(output_spacing)
    for entry in files:
        target = checkout / entry['target']
        if entry['target'] == PRODUCER:
            regular(target)
            writable(target)
            require(digest(target) == BASE_SHA256,
                    'Producer is not the exact pinned baseline; modified/already patched/wrong base refused')
        else:
            require(not os.path.lexists(target), f'Existing helper/test would be overwritten: {target}')
    # Refuse either mode's helper even if not part of the selected output spacing.
    for name in ('support_safe_finalize.py', 'frozen_support_safe_finalize.py'):
        require(not os.path.lexists(folder / name), f'Conflicting mode or helper: {folder / name}')
    return checkout, manifest, files


def restore(checkout):
    """Restore only exact original/installed bytes; retain the original backup."""
    checkout = Path(checkout).absolute()
    real_directory(checkout)
    real_directory(checkout / 'spiral-fitting')
    control = checkout / CONTROL
    real_directory(control)
    regular(control / 'manifest.json')
    record = json.loads((control / 'manifest.json').read_text())
    require(record['schema'] == 2 and record['base_sha256'] == BASE_SHA256, 'Unsupported recovery record')
    require(Path(record['checkout']).resolve() == checkout.resolve(), 'Recovery record belongs to another checkout')
    backup = control / 'original-grow_track_graph.py'
    regular(backup)
    require(digest(backup) == BASE_SHA256, 'Original backup damaged')
    _, expected_files = package_files(record['output_spacing'])
    require(record['files'] == expected_files, 'Recovery target list differs from the validated package')
    writable(checkout)
    writable(checkout / 'spiral-fitting')
    # Validate every path before any removal or replacement.
    for entry in record['files']:
        target = checkout / entry['target']
        if not os.path.lexists(target):
            require(entry['target'] != PRODUCER, 'Producer missing; manual recovery required')
            continue
        regular(target)
        allowed = {entry['sha256'], BASE_SHA256} if entry['target'] == PRODUCER else {entry['sha256']}
        require(digest(target) in allowed, f'User modification preserved; refusing restore: {target}')
    target = checkout / PRODUCER
    if digest(target) != BASE_SHA256:
        staged = control / ('restore-' + uuid.uuid4().hex)
        shutil.copyfile(backup, staged)
        os.chmod(staged, record['original_mode'])
        os.replace(staged, target)
    for entry in record['files']:
        if entry['target'] != PRODUCER:
            target = checkout / entry['target']
            if target.exists():
                require(digest(target) == entry['sha256'], 'Target changed during restore; stopped')
                target.unlink()
    record['state'] = 'RESTORED'
    save_json(control / 'manifest.json', record)
    archived = checkout / (CONTROL + '-restored-' + uuid.uuid4().hex)
    control.rename(archived)
    return {'status': 'RESTORED', 'backup_and_receipt': str(archived)}


def install(checkout, output_spacing, apply=False):
    checkout, manifest, files = preflight(checkout, output_spacing)
    if not apply:
        return {'status': 'CHECKED_NO_WRITES', 'output_spacing': output_spacing, 'resample_spacing': 5, 'units': 'input-volume voxels', 'base_sha256': BASE_SHA256,
                'files': files, 'note': 'Checks the producer file, not the entire checkout commit or dependencies.'}
    control = checkout / CONTROL
    control.mkdir(mode=0o700)  # Exclusive local lock; never reuse an unknown record.
    source = checkout / PRODUCER
    original_mode = stat.S_IMODE(source.stat().st_mode)
    record = {'schema': 2, 'state': 'PREPARED', 'checkout': str(checkout), 'output_spacing': output_spacing,
              'resample_spacing': 5, 'units': 'input-volume voxels', 'base_sha256': BASE_SHA256, 'upstream_commit': manifest['upstream_commit'],
              'original_mode': original_mode, 'files': files}
    backup = control / 'original-grow_track_graph.py'
    shutil.copyfile(source, backup)
    require(digest(backup) == BASE_SHA256, 'Source changed while preparing backup; nothing installed')
    save_json(control / 'manifest.json', record)
    try:
        staged_files = []
        for index, entry in enumerate(files):
            staged = control / f'payload-{index}'
            shutil.copyfile(PACKAGE / entry['source'], staged)
            require(digest(staged) == entry['sha256'], 'Package changed during staging')
            os.chmod(staged, original_mode if entry['target'] == PRODUCER else 0o644)
            staged_files.append((entry, staged))
        # Recheck all destinations after staging, before publishing anything.
        require(digest(source) == BASE_SHA256 and not source.is_symlink(), 'Source changed; installation aborted')
        for entry, _ in staged_files:
            if entry['target'] != PRODUCER:
                require(not os.path.lexists(checkout / entry['target']), 'Destination appeared during staging')
        # New helpers use atomic exclusive links. Replace producer last.
        for entry, staged in staged_files:
            if entry['target'] != PRODUCER:
                os.link(staged, checkout / entry['target'])
        regular(source)
        require(digest(source) == BASE_SHA256, 'Source changed during installation')
        entry, staged = next(pair for pair in staged_files if pair[0]['target'] == PRODUCER)
        os.replace(staged, source)
        require(all(digest(checkout / f['target']) == f['sha256'] for f in files), 'Installed verification failed')
        record['state'] = 'INSTALLED'
        save_json(control / 'manifest.json', record)
    except Exception as error:
        try:
            recovery = restore(checkout)
        except Exception as rollback_error:
            raise InstallError(f'{error}; automatic rollback refused: {rollback_error}. '
                               f'Backup and recovery record retained at {control}') from error
        raise InstallError(f'{error}; rolled back safely, receipt: {recovery["backup_and_receipt"]}') from error
    return {'status': 'INSTALLED', 'output_spacing': output_spacing, 'resample_spacing': 5,
            'units': 'input-volume voxels', 'receipt': str(control / 'manifest.json'),
            'restore': 'Run this tool with --checkout the-same-checkout --restore'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True, help='Existing official villa checkout; never fetched')
    parser.add_argument('--output-spacing', type=int, choices=(5, 10), help='Support-mode output grid spacing in input-volume voxels; required for check/apply')
    parser.add_argument('--resample-spacing', type=int, choices=(5,), default=5, help='Working raster spacing in input-volume voxels (only 5 is supported)')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', '--dry-run', action='store_true', help='Read-only preflight (the default)')
    actions.add_argument('--apply', action='store_true', help='Install exact vendored files, keeping original backup')
    actions.add_argument('--restore', action='store_true', help='Restore exact original and retain backup/receipt')
    args = parser.parse_args(argv)
    if not args.restore and args.output_spacing is None:
        parser.error('--output-spacing is required for check/apply')
    if args.restore and args.output_spacing is not None:
        parser.error('--restore reads its output spacing from the receipt; omit --output-spacing')
    try:
        result = restore(args.checkout) if args.restore else install(args.checkout, args.output_spacing, args.apply)
    except (InstallError, OSError, ValueError, KeyError) as error:
        print(f'Refused: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
