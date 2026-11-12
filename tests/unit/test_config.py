import os
import tempfile
import unittest
from pathlib import Path

from src.utils.config import ConfigError, load_config


class TestConfigLoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.configs_dir = Path(self.tmpdir.name)

        (self.configs_dir / "base.yaml").write_text(
            "data:\n  root: ${EICU_DATA_ROOT}\n"
            "run:\n  mode: base\n  seed: 42\n"
            "federated:\n  rounds: 20\n"
        )
        (self.configs_dir / "debug.yaml").write_text(
            "run:\n  mode: debug\n"
            "federated:\n  rounds: 2\n"
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_merge_overrides_scalar(self):
        cfg = load_config(str(self.configs_dir / "debug.yaml"))
        self.assertEqual(cfg.get("run.mode"), "debug")
        self.assertEqual(cfg.get("federated.rounds"), 2)
        # base-only key should survive the merge untouched
        self.assertEqual(cfg.get("run.seed"), 42)

    def test_env_var_resolution(self):
        os.environ["EICU_DATA_ROOT"] = "/tmp/fake_eicu_root"
        try:
            cfg = load_config(str(self.configs_dir / "debug.yaml"))
            self.assertEqual(cfg.data_root, "/tmp/fake_eicu_root")
        finally:
            del os.environ["EICU_DATA_ROOT"]

    def test_unresolved_env_var_raises(self):
        os.environ.pop("EICU_DATA_ROOT", None)
        cfg = load_config(str(self.configs_dir / "debug.yaml"))
        with self.assertRaises(ConfigError):
            _ = cfg.data_root

    def test_require_missing_key_raises(self):
        cfg = load_config(str(self.configs_dir / "debug.yaml"))
        with self.assertRaises(ConfigError):
            cfg.require("cohort.sepsis.organ_dysfunction.sofa_like_threshold")

    def test_missing_config_file_raises(self):
        with self.assertRaises(ConfigError):
            load_config(str(self.configs_dir / "does_not_exist.yaml"))


if __name__ == "__main__":
    unittest.main()
