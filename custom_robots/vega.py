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
from mani_skill.utils.structs.pose import Pose

from copy import deepcopy


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
            qpos=np.array(
                [
                    0.,  # head j1
                    0,  # arm_j1
                    0.,  # head j2
                    -np.pi/2,  # arm j2
                    0.,  # head j3
                    -np.pi/2,  # arm j3
                    -np.pi/2,  # arm j4
                    0,  # arm j5
                    0,  # arm j6
                    0,  # arm j7
                ] + [0] * 11  # zeros for finger joints
            ),
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
        self.palm_link_name = "R_hand_base"


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
        self.palm = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )


    @property
    def _controller_configs(self):
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joints,
            -0.1,
            0.1,
            stiffness=1e2,
            damping=20,
            force_limit=300,
            use_delta=True,
            normalize_action=True
        )

        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joints,
            None,
            None,
            stiffness=1000,
            damping=1000,
            force_limit=300,
            normalize_action=False
        )

        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.right_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=100,
            damping=20,
            force_limit=300,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
            normalize_action=True,
        )

        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.right_arm_joints,
            pos_lower=-0.02,
            pos_upper=0.02,
            stiffness=200,
            damping=80,
            force_limit=100,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
            normalize_action=False,
        )

        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.right_arm_joints,
            pos_lower=None,
            pos_upper=None,
            rot_lower=None,
            rot_upper=None,
            stiffness=120,
            damping=50,
            force_limit=100,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        hand_pd_joint_pos = PDJointPosControllerConfig(
            self.right_finger_joints,
            lower=None,
            upper=None,
            stiffness=0,
            damping=0,
            force_limit=0,
            normalize_action=False
        )

        hand_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_finger_joints,
            lower=-0.001,
            upper=0.001,
            stiffness=5,
            damping=20,
            force_limit=20,
            use_delta=True,
        )

        # dont move the head
        head_pd_joint_pos = PDJointPosControllerConfig(
            self.head_joints,
            0,
            0,
            0,
            0,
            0,
            use_delta=False
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
            ),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, hand=hand_pd_joint_pos, head=head_pd_joint_pos),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, hand=hand_pd_joint_pos, head=head_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, hand=hand_pd_joint_pos),
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
        """Robot tcp pose, which in this case is the middle finger first link"""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def palm_pose(self):
        """Robot palm pose, which in this case is the right hand base link"""
        return self.palm.pose

    @property
    def finger_tip_pos(self):
        """
        Get the finger tip positions for the right hand
        """
        return torch.stack(
            [fingertip.pose.p for fingertip in self.finger_tips], dim=1
        )

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :10]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    # def reset(self):
    #     super().reset(self.keyframes["rest"].qpos)

