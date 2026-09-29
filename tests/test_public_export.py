"""Synthetic regressions for the fail-closed source-only export."""

from pathlib import Path
import tempfile
import unittest

from scripts.build_source_release import collect_public_files, export_snapshot, validate_public_path


class PublicExportTests(unittest.TestCase):
    def test_release_allowlist(self):
        files = collect_public_files()
        self.assertIn("README.md", files)
        self.assertIn("src/models/gnn_surrogate.py", files)
        self.assertTrue(all(not name.startswith(".git/") for name in files))

    def test_data_and_traversal_paths_rejected(self):
        for name in ("data/network.inp", "src/network.inp", "src/results.csv",
                     "configs/graph.json", "src/checkpoint.pt", "docs/map.png",
                     "../README.md", "src/../../README.md", "C:/README.md",
                     ".git/config", "src/.git/test.py", "src//test.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_public_path(name)

    def test_unlisted_files_and_history_never_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "working"
            root.mkdir()
            (root / "PUBLIC_FILES.txt").write_text("README.md\n", encoding="utf-8")
            (root / "README.md").write_text("Synthetic documentation only.\n", encoding="utf-8")
            (root / "private.inp").write_text("Synthetic excluded fixture.", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("Synthetic history fixture.", encoding="utf-8")
            output = Path(temp) / "snapshot"
            self.assertEqual(export_snapshot(root, output), 1)
            self.assertEqual({p.name for p in output.iterdir()}, {"README.md", "SHA256SUMS.txt"})
            with self.assertRaises(FileExistsError):
                export_snapshot(root, output)
            with self.assertRaises(ValueError):
                export_snapshot(root, root / "export")

    def test_binary_and_credential_content_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "PUBLIC_FILES.txt").write_text("README.md\n", encoding="utf-8")
            for content in (b"binary\x00fixture", ("ghp_" + "x" * 36).encode()):
                (root / "README.md").write_bytes(content)
                with self.assertRaises(ValueError):
                    collect_public_files(root)


if __name__ == "__main__":
    unittest.main()
