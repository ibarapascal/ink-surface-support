"""Check plan integrity and real completion-line formats without scientific compute."""
import importlib.util
import json
import os
import sys
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'tools/flatten.py'
spec = importlib.util.spec_from_file_location('flatten_plan', MODULE)
flatten = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flatten)


class FlattenPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.villa = self.root / 'villa'
        (self.villa / 'lasagna').mkdir(parents=True)
        p = self.villa / 'lasagna/fit.py'
        p.write_text('# mock source identity, not a scientific fixture\n')
        self.lock = self.root / 'lock.json'
        self.lock.write_text(json.dumps({'files': [{'path': 'lasagna/fit.py', 'sha256': flatten.sha(p)}]}))
        self.source = self.root / 'input.tifxyz'
        self.source.mkdir()
        for axis in 'xyz':
            (self.source / (axis + '.tif')).write_bytes(b'not-a-scientific-tiff')
        (self.source / 'meta.json').write_text(json.dumps({'scale': [0.2, 0.2]}))

    def plan(self):
        return flatten.prepare(self.villa, self.source, self.root / 'run',
                               sys.executable, [100., 200., 300.], self.lock)

    def test_prepare_never_starts_compute_or_translates_coordinates(self):
        with patch.object(flatten.subprocess, 'run', side_effect=AssertionError('unexpected compute')):
            plan = self.plan()
        d = json.loads(plan.read_text())
        self.assertEqual(d['source']['source_step'], 5)
        self.assertEqual(d['origin_global_l2_zyx'], [100, 200, 300])
        self.assertEqual(json.loads((plan.parent / 'input.json').read_text()),
                         {'external_surfaces': [{'path': str(self.source.resolve())}], 'voxel_size_um': 9.6})

    def test_wrong_source_pin_rejected_without_partial_directory(self):
        (self.villa / 'lasagna/fit.py').write_text('different source')
        with self.assertRaisesRegex(ValueError, 'Pinned source mismatch'):
            self.plan()
        self.assertFalse((self.root / 'run').exists())

    def test_existing_run_preserved(self):
        plan = self.plan()
        before = plan.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.plan()
        self.assertEqual(plan.read_bytes(), before)

    def test_changed_input_rejected_before_subprocess(self):
        plan = self.plan()
        (self.source / 'x.tif').write_bytes(b'changed')
        with patch.object(flatten, 'LOCK', self.lock), patch.object(flatten.subprocess, 'run', side_effect=AssertionError('unexpected compute')):
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                flatten.run(plan)

    def test_tampered_lock_rejected(self):
        plan = self.plan()
        d = json.loads(plan.read_text())
        d['source_lock']['files'] = []
        plan.write_text(json.dumps(d))
        with patch.object(flatten, 'LOCK', self.lock):
            with self.assertRaisesRegex(ValueError, 'Source lock differs'):
                flatten.run(plan)

    def test_non_native_scale_rejected(self):
        (self.source / 'meta.json').write_text('{"scale":[0.2,0.1]}')
        with self.assertRaisesRegex(ValueError, 'equal positive'):
            self.plan()
        self.assertFalse((self.root / 'run').exists())

    def test_real_minute_and_second_completion_formats(self):
        text = '\n'.join(f"[optimizer] stage{i}: params=map_flatten_ms steps=1000\n[optimizer] stage{i} 'flatten' complete in {t}s" for i,t in enumerate(['71.13','1m10.58','1m10.51']))
        self.assertTrue(flatten.parse_completion(text)['three_fixed_stages'])
        self.assertFalse(flatten.parse_completion(text.replace('steps=1000', 'steps=10', 1))['three_fixed_stages'])
        self.assertFalse(flatten.parse_completion(text + text)['three_fixed_stages'])

    def fake_cli(self, *, exit_code=0, malformed_meta=False,
                 checkpoint=True, final_iterations=True):
        """A stdlib fixture, never the scientific consumer or valid scientific TIFFs."""
        config = self.villa / 'lasagna/configs/flatten_fast.json'
        config.parent.mkdir(exist_ok=True)
        config.write_text('{"fixture":"not a scientific recipe"}')
        code = """import json, os, pathlib, sys
assert sys.argv[3:5] == ['--device', 'cpu']
assert sys.argv[5] == '--out-dir'
# argv[1] is config; argv[2] is sidecar. Actual list checked by parent test.
dest = pathlib.Path(sys.argv[-1])
target = dest/'tifxyz/flatten.tifxyz'
target.mkdir(parents=True)
for a in 'xyz': (target/(a+'.tif')).write_bytes(b'fake-wrapper-fixture')
META
CHECKPOINT
(dest/'observed.json').write_text(json.dumps({'argv':sys.argv,
    'env':{k:os.environ.get(k) for k in ENV_KEYS}, 'pgid':os.getpgrp() if hasattr(os,'getpgrp') else None}))
print('source_step=5 output_step=5')
for i, duration in enumerate(['1m10.51', '1.20', '2.50']):
    print(f'[optimizer] stage{i}: params=map_flatten_ms steps=1000')
    ITERATIONS
    print(f"[optimizer] stage{i} 'flatten' complete in {duration}s")
sys.exit(EXIT)
"""
        code = code.replace('META', "(target/'meta.json').write_text('{bad')" if malformed_meta
                            else "(target/'meta.json').write_text(json.dumps({'scale':[.2,.2]}))")
        code = code.replace('CHECKPOINT', "(dest/'model_final.pt').write_bytes(b'fake checkpoint')" if checkpoint else 'pass')
        code = code.replace('ITERATIONS', "print(f'stage{i} 1000/1000 fixture')" if final_iterations else 'pass')
        code = code.replace('EXIT', str(exit_code)).replace('ENV_KEYS', repr(list(flatten.ENV)))
        fit = self.villa / 'lasagna/fit.py'
        fit.write_text(code)
        self.lock.write_text(json.dumps({'files': [
            {'path':'lasagna/fit.py','sha256':flatten.sha(fit)},
            {'path':'lasagna/configs/flatten_fast.json','sha256':flatten.sha(config)}]}))

    def execute_fake(self, **options):
        self.fake_cli(**options)
        path = self.plan()
        with patch.object(flatten, 'LOCK', self.lock):
            status = flatten.run(path)
        return status, path.parent, json.loads((path.parent/'result.json').read_text())

    def test_real_fake_subprocess_argv_env_minutes_and_inherited_group(self):
        status, root, result = self.execute_fake()
        self.assertEqual(status, 0)
        self.assertEqual(result['state'], 'CLI_COMPLETE_BASIC_OUTPUT_CHECKS')
        self.assertTrue(result['checks']['final_iterations_reached'])
        observed = json.loads((root/'consumer/observed.json').read_text())
        self.assertEqual(observed['argv'][1:], [str(self.villa/'lasagna/configs/flatten_fast.json'),
                         str(root/'input.json'), '--device', 'cpu', '--out-dir', str(root/'consumer')])
        for key, value in flatten.ENV.items():
            self.assertEqual(observed['env'][key], value)
        if hasattr(os, 'getpgrp'):
            self.assertEqual(observed['pgid'], os.getpgrp())
        before = (root/'result.json').read_bytes()
        with patch.object(flatten, 'LOCK', self.lock):
            with self.assertRaisesRegex(ValueError, 'already started'):
                flatten.run(root/'plan.json')
        self.assertEqual((root/'result.json').read_bytes(), before)

    def test_nonzero_fake_cli_never_claims_complete(self):
        status, _, result = self.execute_fake(exit_code=7)
        self.assertEqual(status, 2)
        self.assertEqual(result['cli_exit_code'], 7)
        self.assertEqual(result['state'], 'INCOMPLETE_OR_FAILED')

    def test_malformed_output_meta_has_failure_receipt(self):
        status, _, result = self.execute_fake(malformed_meta=True)
        self.assertEqual(status, 2)
        self.assertEqual(result['cli_exit_code'], 0)
        self.assertIn('JSONDecodeError', result['error'])
        self.assertEqual(result['state'], 'INCOMPLETE_OR_FAILED')

    def test_missing_final_iteration_not_complete(self):
        status, _, result = self.execute_fake(final_iterations=False)
        self.assertEqual(status, 2)
        self.assertFalse(result['checks']['final_iterations_reached'])

    def test_missing_checkpoint_not_complete(self):
        status, _, result = self.execute_fake(checkpoint=False)
        self.assertEqual(status, 2)
        self.assertFalse(result['checkpoint_present'])

    def test_spawn_failure_has_started_and_failure_receipts(self):
        plan = self.plan()
        with patch.object(flatten, 'LOCK', self.lock), patch.object(flatten.subprocess, 'run', side_effect=OSError('fixture spawn failure')):
            self.assertEqual(flatten.run(plan), 2)
        result = json.loads((plan.parent/'result.json').read_text())
        self.assertIn('fixture spawn failure', result['error'])
        self.assertTrue((plan.parent/'started.json').exists())
        self.assertEqual(result['state'], 'INCOMPLETE_OR_FAILED')

    def test_sidecar_and_plan_hash_edit_cannot_change_recipe(self):
        plan = self.plan()
        side = plan.parent/'input.json'
        value = json.loads(side.read_text());value['voxel_size_um'] = 2.4
        side.write_text(json.dumps(value))
        data = json.loads(plan.read_text());data['sidecar_sha256'] = flatten.sha(side)
        plan.write_text(json.dumps(data))
        with patch.object(flatten, 'LOCK', self.lock), patch.object(flatten.subprocess, 'run', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError, 'noncanonical sidecar'):
                flatten.run(plan)

    def test_dangling_reserved_output_is_not_followed(self):
        plan = self.plan()
        (plan.parent/'consumer').symlink_to(self.root/'outside')
        with patch.object(flatten, 'LOCK', self.lock):
            with self.assertRaisesRegex(ValueError, 'output path reserved'):
                flatten.run(plan)
        self.assertFalse((self.root/'outside').exists())

    def test_prepare_rejects_nested_source_output_and_dangling_run(self):
        with self.assertRaisesRegex(ValueError, 'outside the input'):
            flatten.prepare(self.villa, self.source, self.source/'run', sys.executable,
                            [0,0,0], self.lock)
        (self.root/'run').symlink_to(self.root/'outside')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.plan()
        self.assertFalse((self.root/'outside').exists())

    def test_python_path_lookup_preserves_venv_symlink(self):
        folder = self.root/'bin';folder.mkdir()
        executable = folder/'python3';executable.symlink_to(sys.executable)
        with patch.dict(os.environ, {'PATH':str(folder)}):
            plan = flatten.prepare(self.villa, self.source, self.root/'run', 'python3', [0,0,0], self.lock)
        self.assertEqual(json.loads(plan.read_text())['python'], str(executable.absolute()))

    def test_non_executable_python_rejected_before_directory_creation(self):
        executable = self.root/'not-python';executable.write_text('not executable');executable.chmod(0o644)
        for name in (str(executable), str(self.root/'missing')):
            with self.assertRaisesRegex(ValueError, 'executable'):
                flatten.prepare(self.villa, self.source, self.root/'run', name, [0,0,0], self.lock)
        self.assertFalse((self.root/'run').exists())

    def test_prepare_failure_optional_receipt_without_partial_run_dir(self):
        (self.villa/'lasagna/fit.py').write_text('changed')
        receipt = self.root/'failure.json'
        with patch.object(flatten, 'LOCK', self.lock), contextlib.redirect_stderr(io.StringIO()):
            status = flatten.main(['prepare','--villa',str(self.villa),'--source',str(self.source),
                '--run-dir',str(self.root/'run'),'--origin-zyx','0','0','0','--receipt',str(receipt)])
        self.assertEqual(status,2)
        self.assertEqual(json.loads(receipt.read_text())['state'],'PREPARE_REJECTED')
        self.assertFalse((self.root/'run').exists())

    def test_postrun_source_change_retains_failed_receipt(self):
        self.fake_cli()
        fit = self.villa/'lasagna/fit.py'
        code = fit.read_text().replace('sys.exit(0)', "pathlib.Path(__file__).write_text('changed after launch')\nsys.exit(0)")
        fit.write_text(code)
        lock = json.loads(self.lock.read_text());lock['files'][0]['sha256'] = flatten.sha(fit)
        self.lock.write_text(json.dumps(lock))
        plan = self.plan()
        with patch.object(flatten, 'LOCK', self.lock):
            self.assertEqual(flatten.run(plan), 2)
        result = json.loads((plan.parent/'result.json').read_text())
        self.assertEqual(result['cli_exit_code'], 0)
        self.assertEqual(result['state'], 'INCOMPLETE_OR_FAILED')
        self.assertIn('Pinned source mismatch', result['error'])

    def test_pinned_source_symlink_cannot_redirect_import_root(self):
        fit = self.villa/'lasagna/fit.py'
        outside = self.root/'elsewhere.py'
        outside.write_bytes(fit.read_bytes())
        fit.unlink();fit.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Pinned source mismatch'):
            self.plan()
        self.assertFalse((self.root/'run').exists())

    def test_repository_lock_includes_config_and_root_fiber_source(self):
        lock = json.loads(flatten.LOCK.read_text())
        paths = [x['path'] for x in lock['files']]
        self.assertEqual(len(paths), 94)
        self.assertEqual(len(set(paths)), 94)
        self.assertIn('vesuvius/src/vc3d_fiber_format/__init__.py', paths)
        self.assertNotIn('lasagna/vesuvius/src/vc3d_fiber_format/__init__.py', paths)
        self.assertIn('lasagna/configs/flatten_fast.json', paths)



if __name__ == '__main__':
    unittest.main()
