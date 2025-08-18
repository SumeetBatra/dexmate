import sapien
import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe, DictControllerConfig
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils import sapien_utils
from typing import Dict, Union, List


@register_agent()
class VegaUpperBody(BaseAgent):
    uid = "vega_upper_body"
    urdf_path = "dexmate-urdf/robots/humanoid/vega_1/vega_upper_body_right_arm_maniskill.urdf"
    # set the frictions on the right fingertips to be higher to avoid slipping
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0),
        ),
        # finger tips
        link={
            "R_th_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "R_ff_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "R_mf_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "R_rf_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "R_lf_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        }
    )
    # specify some initial resting config of the robot
    keyframes = dict(
        rest=Keyframe(
            qpos=np.zeros(21),
            pose=sapien.Pose()
        )
    )

    initial_pose: sapien.Pose = sapien.Pose(p=[0, 0, 1], q=[1, 0, 0, 0])

    def __init__(self, *args, **kwargs):
        self.right_arm_joints = [
            "R_arm_j1",
            "R_arm_j2",
            "R_arm_j3",
            "R_arm_j4",
            "R_arm_j5",
            "R_arm_j6",
            "R_arm_j7",
        ]

        self.right_finger_joints = [
            "R_th_j0",
            "R_ff_j1",
            "R_mf_j1",
            "R_rf_j1",
            "R_lf_j1",
            "R_th_j1",
            "R_ff_j2",
            "R_mf_j2",
            "R_rf_j2",
            "R_lf_j2",
            "R_th_j2"
        ]


        self.head_joints = [
            "head_j1",
            "head_j2",
            "head_j3"
        ]

        self.joint_names = self.right_arm_joints + self.right_finger_joints + self.head_joints

        self.finger_tip_names = [
            "R_th_tip",
            "R_ff_tip",
            "R_mf_tip",
            "R_rf_tip",
            "R_lf_tip"
        ]

        self.finger_link_names = [
            "R_th_l2",
            "R_ff_l2",
            "R_mf_l2",
            "R_rf_l2",
            "R_lf_l2"
        ]

        # use the right palm + y offset as the tcp
        self.ee_link_name = "R_hand_base"


        super(VegaUpperBody, self).__init__(*args, **kwargs)

    def _after_init(self):
        # track the right hand finger links and finger tips
        self.finger_tips = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.finger_tip_names
        )
        self.finger_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.finger_link_names
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )


    @property
    def _controller_configs(self):
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joints,
            -0.1,
            0.1,
            stiffness=1e2,
            damping=15,
            force_limit=200,
            use_delta=True,
        )

        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joints,
            None,
            None,
            stiffness=1000,
            damping=200,
            force_limit=100,
            normalize_action=False
        )

        hand_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.right_finger_joints,
            None,
            None,
            stiffness=400,
            damping=10,
            force_limit=50,
            mimic={
                "R_th_j2": {"joint": "R_ff_j2"},
                "R_mf_j2": {"joint": "R_ff_j2"},
                "R_rf_j2": {"joint": "R_ff_j2"},
                "R_lf_j2": {"joint": "R_ff_j2"},
            }
        )

        # dont move the head
        head_pd_joint_pos = PDJointPosControllerConfig(
            self.head_joints,
            0,
            0,
            1e3,
            1e2,
            100,
            use_delta=True
        )

        controller_config = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                hand=hand_pd_joint_pos,
                head=head_pd_joint_pos
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                hand=hand_pd_joint_pos,
                head=head_pd_joint_pos
            )
        )
        return controller_config

    def get_proprioception(self):
        obs = super().get_proprioception()
        return obs

    def is_grasping(self, object: Actor, min_force=0.5, min_contacts=5):
        """
        Check if the robot is grasping the given object by checking contact forces
        :param object: the object to check grasping against
        :param min_force: the minimum contact force to check
        :param min_contacts: number of contacts i.e. fingertips that are in contact with the object
        By setting min_contacts to 5, we assume that all fingertips are in contact with the object
        means a successful grasp. This simplifies the is_grasping logic.
        """
        num_envs = object.angular_velocity.shape[0]
        valid_contacts = torch.zeros((num_envs,)).to(self.tcp_pose.device)
        for fingertip in self.finger_tips:
            contact_forces = self.scene.get_pairwise_contact_forces(
                fingertip, object
            )

            if contact_forces is None or len(contact_forces) == 0:
                continue

            force_mag = torch.linalg.norm(contact_forces, dim=1)
            valid_contacts[torch.where(force_mag >= min_force)[0]] += 1

        return valid_contacts >= min_contacts

    @property
    def tcp_pos(self):
        """Robot tcp pose, which in this case is the right palm"""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :10]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

