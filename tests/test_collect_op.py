# filepath: /home/ruoyu/dev/pfd-kit/tests/test_collect_data_op.py
import os
import unittest
import tempfile
from pathlib import Path
from contextlib import contextmanager

from ase import Atoms
from ase.io import write, read

from pfd.op.collect import CollectData
from dflow.python import OPIO


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def make_extxyz(path: Path, n_frames: int, symbol: str = "H"):
    """Create an extxyz file with n_frames single-atom structures."""
    frames = []
    for i in range(n_frames):
        at = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
        at.info["frame_id"] = i
        frames.append(at)
    write(path, frames)  # ASE infers format from extension


class TestCollectDataOP(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.work = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_basic_split(self):
        """
        No previous structures, no optional_structures.
        Expect:
          - *_all.extxyz has all original frames
          - *_train.extxyz + *_test.extxyz partition the originals
          - Default test_size=0.1 (may round) so we control by setting test_size.
        """
        with pushd(self.work):
            # Prepare two input files totalling 20 frames
            f1 = Path("iter_a.extxyz")
            f2 = Path("iter_b.extxyz")
            make_extxyz(f1, 8)
            make_extxyz(f2, 12)

            op = CollectData()
            out = op.execute(
                OPIO(
                    {
                        "structures": [f1, f2],
                        "optional_parameters": {"test_size": 0.25, "structures": "structures"},
                    }
                )
            )

            train_file = Path(out["structures"])
            all_file = Path(out["iter_structures"])
            test_file = Path(out["test_structures"])

            self.assertTrue(train_file.is_file(), "Train structures file missing")
            self.assertTrue(all_file.is_file(), "All structures file missing")
            self.assertTrue(test_file.is_file(), "Test structures file missing")

            all_structs = read(all_file, index=":")
            train_structs = read(train_file, index=":")
            test_structs = read(test_file, index=":")

            self.assertEqual(len(all_structs), 20)
            self.assertEqual(len(train_structs) + len(test_structs), 20)
            # With 25% test_size on 20 frames, expect 5 test frames
            self.assertEqual(len(test_structs), 5)

    def test_with_previous_iteration(self):
        """
        Provide pre_structures. Output *_all.extxyz should contain previous + current.
        """
        with pushd(self.work):
            prev = Path("prev.extxyz")
            curr1 = Path("curr1.extxyz")
            curr2 = Path("curr2.extxyz")
            make_extxyz(prev, 5)
            make_extxyz(curr1, 4)
            make_extxyz(curr2, 6)

            op = CollectData()
            out = op.execute(
                OPIO(
                    {
                        "structures": [curr1, curr2],
                        "pre_structures": [prev],
                        "optional_parameters": {"test_size": 0.2, "structures": "structures"},
                    }
                )
            )

            all_file = Path(out["iter_structures"])
            train_file = Path(out["structures"])
            test_file = Path(out["test_structures"])

            all_structs = read(all_file, index=":")
            # pre_structures_ls = previous + current iteration
            self.assertEqual(len(all_structs), 5 + 4 + 6)

            train_structs = read(train_file, index=":")
            test_structs = read(test_file, index=":")
            # Train should include previous + a subset of current (since previous appended after split)
            self.assertGreaterEqual(len(train_structs), 5)  # at least previous ones retained
            self.assertGreater(len(train_structs), len(test_structs))

    def test_iter_id_labeling(self):
        """
        iter_id should label each current Atoms object with info['iter'] before writing.
        Verify by reading *_all.extxyz (which, without pre_structures, equals current originals).
        """
        with pushd(self.work):
            curr = Path("curr.extxyz")
            make_extxyz(curr, 3)

            op = CollectData()
            out = op.execute(
                OPIO(
                    {
                        "structures": [curr],
                        "optional_parameters": {"test_size": 0.34, "structures": "structures"},
                        "iter_id": "ITER_5",
                    }
                )
            )
            all_file = Path(out["iter_structures"])
            structs = read(all_file, index=":")
            self.assertTrue(all("iter" in at.info for at in structs))
            self.assertTrue(all(at.info["iter"] == "ITER_5" for at in structs))

    @unittest.expectedFailure
    def test_optional_structures_current_bug(self):
        """
        Current implementation sets test_structures to list[Path] (not Atoms) when optional_structures
        is provided, causing ase.io.write to likely fail.
        This test documents the behavior; once fixed, remove expectedFailure and add assertions.
        """
        with pushd(self.work):
            curr = Path("curr.extxyz")
            opt = Path("opt.extxyz")
            make_extxyz(curr, 4)
            make_extxyz(opt, 2)

            op = CollectData()
            op.execute(
                OPIO(
                    {
                        "structures": [curr],
                        "optional_structures": [opt],
                        "optional_parameters": {"structures": "structures"},
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()