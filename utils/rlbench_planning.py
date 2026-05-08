# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    EndEffectorPoseViaIK, 
    Scene,
)

class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        super().action(scene, action, ignore_collisions)

class EndEffectorPoseViaIK2(EndEffectorPoseViaIK):
    def action(self, scene: Scene, action, ignore_collisions=True):
        # ignore_collisions is unused for IK, but must be accepted
        return super().action(scene, action)

    def record_start(self, scene: Scene, **kwargs):
        pass

    def record_end(self, scene: Scene, **kwargs):
        pass
