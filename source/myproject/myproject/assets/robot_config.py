# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from myproject.assets import ISAAC_ASSET_DIR


MY_ROBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_ASSET_DIR}/robots/poclegs.usd",
        usd_path=f"{ISAAC_ASSET_DIR}/robots/mod_poclegs_2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            'LL_HR': 0.0,
            'LR_HR': 0.0,
            'LL_HAA': -0.1745,
            'LR_HAA': -0.1745, # 10 degrees
            'LL_HFE': -0.1745,
            'LR_HFE': -0.1745,
            'LL_KFE': 0.3491,
            'LR_KFE': 0.3491, # 20 degrees
            'LL_FFE': -0.1745, # 10 degrees
            'LR_FFE': -0.1745 # 10 degrees
        },
    ),
    actuators={
        "hr": DelayedPDActuatorCfg(
            joint_names_expr=[".*HR"],
            effort_limit=24.0,
            velocity_limit=23.0,
            stiffness=10.0,
            damping=1.5,
            armature=6.9e-5 * 81,
            friction=0.02,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "haa": DelayedPDActuatorCfg(
            joint_names_expr=[".*HAA"],
            effort_limit=30.0,
            velocity_limit=15.0,
            stiffness=15.0,
            damping=1.5,
            armature=9.4e-5 * 81,
            friction=0.02,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "kfe": DelayedPDActuatorCfg(
            joint_names_expr=[".*HFE", ".*KFE"],
            effort_limit=30.0,
            velocity_limit=20.0,
            stiffness=15.0,
            damping=1.5,
            armature=1.5e-4 * 81,
            friction=0.02,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "ffe": DelayedPDActuatorCfg(
            joint_names_expr=[".*FFE"],
            effort_limit=20.0,
            velocity_limit=23.0,
            stiffness=10.0,
            damping=1.5,
            armature=6.9e-5 * 81,
            friction=0.02,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
