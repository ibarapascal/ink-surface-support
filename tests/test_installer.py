"""Installer tests using synthetic temporary checkout layouts."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('offline_installer', ROOT / 'tools/install_patch.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkout = Path(self.tmp.name).resolve() / 'villa'
        (self.checkout / 'spiral-fitting').mkdir(parents=True)
        self.base = self.checkout / tool.PRODUCER
        shutil.copyfile(ROOT / 'vendor/villa-pinned/grow_track_graph.py', self.base)
        os.chmod(self.base, 0o755)

    def test_default_check_writes_nothing(self):
        before = sorted(p.relative_to(self.checkout) for p in self.checkout.rglob('*'))
        result = tool.install(self.checkout, 10)
        self.assertEqual(result['status'], 'CHECKED_NO_WRITES')
        self.assertEqual(before, sorted(p.relative_to(self.checkout) for p in self.checkout.rglob('*')))
        self.assertEqual(tool.digest(self.base), tool.BASE_SHA256)

    def test_both_output_spacings_exact_install_restore_and_switch(self):
        for mode in (10, 5):
            result = tool.install(self.checkout, mode, apply=True)
            self.assertEqual(result['status'], 'INSTALLED')
            _, files = tool.package_files(mode)
            for entry in files:
                self.assertEqual(tool.digest(self.checkout / entry['target']), entry['sha256'])
            self.assertEqual(self.base.stat().st_mode & 0o777, 0o755)
            record = json.loads((self.checkout / tool.CONTROL / 'manifest.json').read_text())
            self.assertEqual(record['state'], 'INSTALLED')
            self.assertEqual(tool.digest(self.checkout / tool.CONTROL / 'original-grow_track_graph.py'), tool.BASE_SHA256)
            tool.restore(self.checkout)
            self.assertEqual(tool.digest(self.base), tool.BASE_SHA256)
            self.assertFalse((self.checkout / tool.CONTROL).exists())
            for entry in files:
                if entry['target'] != tool.PRODUCER:
                    self.assertFalse((self.checkout / entry['target']).exists())

    def test_wrong_or_modified_base_refused_without_writes(self):
        self.base.write_bytes(self.base.read_bytes() + b'\n# user edit\n')
        original = self.base.read_bytes()
        with self.assertRaises(tool.InstallError):
            tool.install(self.checkout, 10, True)
        self.assertEqual(original, self.base.read_bytes())
        self.assertFalse((self.checkout / tool.CONTROL).exists())

    def test_conflicting_helper_or_test_never_overwritten(self):
        for filename in ('support_safe_finalize.py', 'frozen_support_safe_finalize.py',
                         'test_support_safe_finalize.py'):
            p = self.base.parent / filename
            p.write_text('user data')
            with self.assertRaises(tool.InstallError):
                tool.install(self.checkout, 10, True)
            self.assertEqual(p.read_text(), 'user data')
            self.assertEqual(tool.digest(self.base), tool.BASE_SHA256)
            self.assertFalse((self.checkout / tool.CONTROL).exists())
            p.unlink()

    def test_target_symlink_refused(self):
        original = self.base.read_bytes()
        outside = self.checkout.parent / 'outside.py'
        outside.write_bytes(original)
        self.base.unlink()
        self.base.symlink_to(outside)
        with self.assertRaises(tool.InstallError):
            tool.install(self.checkout, 5, True)
        self.assertEqual(outside.read_bytes(), original)

    def test_parent_symlink_refused(self):
        external = self.checkout.parent / 'outside'
        (self.checkout / 'spiral-fitting').rename(external)
        (self.checkout / 'spiral-fitting').symlink_to(external, target_is_directory=True)
        with self.assertRaises(tool.InstallError):
            tool.install(self.checkout, 10, True)

    def test_dangling_helper_symlink_refused(self):
        (self.base.parent / 'support_safe_finalize.py').symlink_to('missing.py')
        with self.assertRaises(tool.InstallError):
            tool.install(self.checkout, 10, True)

    def test_readonly_destination_refused(self):
        os.chmod(self.base, 0o444)
        with self.assertRaises(tool.InstallError):
            tool.install(self.checkout, 10, True)
        self.assertFalse((self.checkout / tool.CONTROL).exists())

    def test_modified_installed_file_blocks_restore_and_preserves_all(self):
        tool.install(self.checkout, 10, True)
        helper = self.base.parent / 'support_safe_finalize.py'
        helper.write_text('local research changes')
        before = self.base.read_bytes()
        with self.assertRaises(tool.InstallError):
            tool.restore(self.checkout)
        self.assertEqual(helper.read_text(), 'local research changes')
        self.assertEqual(self.base.read_bytes(), before)
        self.assertTrue((self.base.parent / 'test_support_safe_finalize.py').exists())

    def test_second_install_and_implicit_mode_switch_refused(self):
        tool.install(self.checkout, 10, True)
        before = self.base.read_bytes()
        for mode in (10, 5):
            with self.assertRaises(tool.InstallError):
                tool.install(self.checkout, mode, True)
            self.assertEqual(self.base.read_bytes(), before)

    def test_mid_install_failure_rolls_back(self):
        real_link = os.link
        calls = []
        def fail_second(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError('injected disk/write failure')
            return real_link(source, target)
        with mock.patch.object(tool.os, 'link', side_effect=fail_second):
            with self.assertRaisesRegex(tool.InstallError, 'rolled back safely'):
                tool.install(self.checkout, 10, True)
        self.assertEqual(tool.digest(self.base), tool.BASE_SHA256)
        self.assertFalse((self.base.parent / 'support_safe_finalize.py').exists())
        self.assertFalse((self.base.parent / 'test_support_safe_finalize.py').exists())
        self.assertTrue(list(self.checkout.glob(tool.CONTROL + '-restored-*')))

    def test_package_hash_failure_precedes_checkout_writes(self):
        package = self.checkout.parent / 'package'
        shutil.copytree(ROOT / 'vendor', package / 'vendor')
        shutil.copytree(ROOT / 'patches', package / 'patches')
        (package / 'vendor/output-spacing-10/support_safe_finalize.py').write_text('changed')
        with mock.patch.object(tool, 'PACKAGE', package):
            with self.assertRaises(tool.InstallError):
                tool.install(self.checkout, 10, True)
        self.assertFalse((self.checkout / tool.CONTROL).exists())

    @unittest.skipUnless(shutil.which('patch'), 'Optional system patch executable absent')
    def test_review_diffs_apply_cleanly_to_exact_base(self):
        # This is an independent review-diff test, not the installation path.
        for mode in (10, 5):
            tmp = self.checkout.parent / ('diff-' + str(mode))
            (tmp / 'spiral-fitting').mkdir(parents=True)
            shutil.copyfile(ROOT / 'vendor/villa-pinned/grow_track_graph.py', tmp / tool.PRODUCER)
            self.assertEqual(tool.digest(tmp / tool.PRODUCER), tool.BASE_SHA256)
            _, files = tool.package_files(mode)
            subprocess.run([shutil.which('patch'), '-p1', '-i', str(ROOT / 'patches' / ('fine-support-output-spacing-' + str(mode) + '.patch'))],
                           cwd=tmp, check=True, capture_output=True, text=True)
            for entry in files:
                self.assertEqual(tool.digest(tmp / entry['target']), entry['sha256'])


if __name__ == '__main__':
    unittest.main()
