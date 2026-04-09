import tempfile
import unittest
from pathlib import Path

from attention_kernel_challenge.sandbox import build_nsjail_command


class SandboxTests(unittest.TestCase):
    def test_nsjail_command_uses_chroot_and_workspace_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "sandbox").mkdir()
            (repo_root / "sandbox" / "nsjail.cfg").write_text("mode: ONCE\n")
            command = build_nsjail_command(
                module="attention_kernel_challenge.cli",
                module_args=["_eval-reference-internal", "--suite", "smoke"],
                repo_root=str(repo_root),
                scratch_dir="/tmp/akc-scratch",
                jail_root="/tmp/akc-root",
            )

        command_text = " ".join(command)
        self.assertIn("--chroot /tmp/akc-root", command_text)
        self.assertIn("--bindmount_ro", command_text)
        self.assertIn(f"{repo_root}:/workspace", command_text)
        self.assertIn("PYTHONNOUSERSITE=1", command_text)


if __name__ == "__main__":
    unittest.main()
