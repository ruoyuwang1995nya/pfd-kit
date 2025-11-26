import unittest
import os
from pfd.exploration.task.ase_task_group import AseTaskGroup
from pfd.exploration.task.stage import ExplorationStage
from ase.build import bulk
from ase.io import write
from pathlib import Path


class TestAseTaskGroup(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path("ase_task_group_test")
        self.work_dir.mkdir(exist_ok=True)
        self.confs_dir = self.work_dir / "confs"
        self.confs_dir.mkdir(exist_ok=True)

        # Create dummy configuration files
        self.init_confs = []
        for i in range(2):
            conf_path = self.confs_dir / f"conf_{i}.extxyz"
            atoms = bulk("Si", "diamond", a=5.43)
            write(conf_path, atoms)
            self.init_confs.append(str(conf_path))

    def tearDown(self):
        import shutil

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def test_make_exploration_stage(self):
        expl_stages_config = [
            [
                {
                    "conf_idx": [0],
                    "n_sample": 1,
                    "temps": [300, 600],
                },
                {"conf_idx": [1], "n_sample": 1, "temps": [400], "press": [1.0]},
            ]
        ]

        expl_stages = []
        for stg_config in expl_stages_config:
            expl_stage = ExplorationStage()
            for task_grp_config in stg_config:
                task_grps = AseTaskGroup.make_task_grp_from_conf(
                    task_grp_config,
                    self.init_confs,
                )
                # for task_grp in task_grps:
                expl_stage.add_task_group(task_grps)
            expl_stages.append(expl_stage)

        self.assertEqual(len(expl_stages), 1)
        stage = expl_stages[0]
        self.assertEqual(len(stage.explor_groups), 2)

        # Check first task group
        tg1 = stage.explor_groups[0]
        self.assertIsInstance(tg1, AseTaskGroup)
        self.assertEqual(tg1.temps, [300, 600])
        self.assertEqual(tg1.press, [None])

        # Check second task group
        tg2 = stage.explor_groups[1]
        self.assertIsInstance(tg2, AseTaskGroup)
        self.assertEqual(tg2.temps, [400])
        self.assertEqual(tg2.press, [1.0])

    def test_make_exploration_stage_multiple_confs(self):
        expl_stages_config = [
            [
                {
                    "conf_idx": [0, 1],
                    "n_sample": 1,
                    "temps": [300],
                }
            ]
        ]

        for stg_config in expl_stages_config:
            expl_stage = ExplorationStage()

            for task_grp_config in stg_config:
                task_grps = AseTaskGroup.make_task_grp_from_conf(
                    task_grp_config,
                    self.init_confs,
                )
                expl_stage.add_task_group(task_grps)

        self.assertEqual(len(expl_stage.explor_groups), 2)
        self.assertIsInstance(expl_stage.explor_groups[0], AseTaskGroup)
        self.assertIsInstance(expl_stage.explor_groups[1], AseTaskGroup)
        self.assertEqual(expl_stage.explor_groups[0].temps, [300])
        self.assertEqual(expl_stage.explor_groups[1].temps, [300])


if __name__ == "__main__":
    unittest.main()
