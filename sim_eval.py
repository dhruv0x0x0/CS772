import gymnasium as gym
import mani_skill.envs
import sim_env.maniskill_env.envs
from sim_env.maniskill_env.wrappers.record_env import RecordEnv
import numpy as np
from mani_skill.utils.structs.pose import Pose,to_sapien_pose
np.set_printoptions(precision=3, suppress=True)
import sapien
import time
import cv2
import sys
import zmq
from dataclasses import dataclass
from einops import rearrange
from sim_env.maniskill_env.utils.object_builder import ObjectId
from sim_env.maniskill_env.utils.scene_builder import SceneId
from tokeniser import TemporalBPEProcessor
import torch
import torch.nn as nn
from vision_model_getter import get_resnet, replace_bn_with_gn
from model import create_model
import os
from typing import Sequence
from policy import generate
from torchvision import transforms
def get_relevant_info(ori_env, initial_pose = np.array([
    [1,0,0,0.097],
    [0,1,0,0],
    [-0.1,0,1,0.378],
    [0,0,0,1]
], dtype=np.float32)
   ):
    """
    Extracts and preprocesses wrist-camera image, relative TCP pose, and gripper state.
    """
    obs = ori_env.get_obs()
    imgs = ori_env.get_sensor_images()
    wrist = imgs['wrist_cam']['rgb'][0].cpu().numpy()
    frame = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    pad = (400 - 224) // 2
    canvas[pad:pad+224, :, :] = frame
    canvas = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA)

    tcp = obs['extra']['tcp_pose'][0]
    grip = obs['agent']['qpos'][0, 7] / 0.04
    T = sapien.Pose(p=tcp[:3], q=tcp[3:]).to_transformation_matrix()
    relT = np.linalg.inv(initial_pose) @ T

    return canvas, relT.astype(np.float32), np.float32(grip)
def get_real_time_updates(obs, initial_pose = np.array([
    [1,0,0,0.097],
    [0,1,0,0],
    [-0.1,0,1,0.378],
    [0,0,0,1]
], dtype=np.float32)
   ):
    """
    Extracts and preprocesses wrist-camera image, relative TCP pose, and gripper state.
    """
    tcp = obs['extra']['tcp_pose'][0]
    grip = obs['agent']['qpos'][0, 7] / 0.04
    T = sapien.Pose(p=tcp[:3], q=tcp[3:]).to_transformation_matrix()
    relT = np.linalg.inv(initial_pose) @ T

    return relT.astype(np.float32), np.float32(grip)
def get_action(offset_pose: np.ndarray=np.eye(4), gripper_state: float=1) -> np.ndarray:
    initial_pose = np.array(
        [
            [1, 0, 0.1, 0.097],
            [0, 1.0, 0.0, 0.0],
            [-0.1, 0.0, 1, 0.378],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_pose = initial_pose @ offset_pose 
    pose = sapien.Pose(target_pose)
    action = np.zeros(8)
    #print(np.rad2deg(pose.get_rpy()))
    #print(f"Input: {pose.get_p()}")
    action[:3] = pose.get_p()
    action[3:7] = pose.get_q()
    action[7] = gripper_state
    return action
from torch.quantization import quantize_dynamic
def load_model(checkpoint_path: str, device: str):
    # build model dict
    model = nn.ModuleDict({
        'vision_encoder': replace_bn_with_gn(
            get_resnet('resnet18')
        ),
        'lldm': create_model(
            vocab_size=500, d_model=768, n_heads=12, n_layers=12
        )
    })
    if os.path.isfile(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    
    model.to(device).eval()
    model = quantize_dynamic(
    model,   # You can add more layer types if needed
    dtype=torch.qint8
)

    return model

class LLDM_Policy:
    def __init__(self, checkpoint = 'model_epoch1_9112.pt', current_obs= None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = TemporalBPEProcessor.load("saved_processor")
        self.model = load_model(checkpoint, self.device)
        #print(current_obs.shape)
        self.last_obs = self.tokenizer(current_obs)
        self.transform = transforms.ToTensor()

    def state_to_tokens(self, state: dict) -> torch.LongTensor:
        tokens = self.tokenizer([state])
        return torch.tensor(tokens[0], dtype=torch.long, device=self.device)

    def tokens_to_state(self, tokens: Sequence[int]) -> dict:
        state = self.tokenizer.decode(tokens)
        return state
    
    def predict_action(self, img, relT, grip):
        proprio = torch.from_numpy(
            np.concatenate([relT.reshape(-1), np.array([grip])])
        ).unsqueeze(0).to(self.device) 
        #print(img.shape)
        img = self.transform(img).unsqueeze(0)#torch.from_numpy(img).permute(2, 0, 1).float().div(255.)
        vis_feats = self.model['vision_encoder'](img)         # e.g. [1, feat_dim]
        obs_feats = torch.cat([vis_feats, proprio], dim=-1) 
        input_ids = self.last_obs[:, -2*17:]   
        #print(input_ids.shape)
        new_tokens = generate(model =self.model['lldm'], prompt= input_ids, cond= obs_feats, mask_id = 0, device= self.device)#logits = self.model['lldm'](input_ids, obs_feats)      # [1, seq_len, vocab]
        next_state = self.tokens_to_state(new_tokens)
        #self.last_pred = next_state.squeeze(0)
        return next_state.squeeze(0)
    
    def update_last_obs(self, next_obs):
        self.last_obs= self.tokenizer(next_obs)

        
        
    


_seed = np.random.randint(0,2**32 - 1)
#_seed = 237284799
# env = gym.make(
#     "PickAnything", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
#     num_envs=1,
#     control_mode="pd_ee_pose_quat", # there is also "pd_joint_delta_pos", ...
#     render_mode="human",
#     robot_uids="floating_gripper_v1",
#     show_meshes=True,
#     enhanced_determinism=True,
#     obs_mode="rgb",
#     sensor_configs = dict(
#         wrist_cam=dict(
#             shader_pack="rt",
#         )
#     ),
#     randomize_along_scene=True,
#     spawn_goal_site=False,
#     #no_randomization=True,
#     upright=True,
#     num_other_objects=10,
# )
from mani_skill.utils import io_utils
json_path = 'monster_row_wise/2025_02_28_18_27_44_PickAnything.json'
json_data = io_utils.load_json(json_path)
env_info = json_data["env_info"]
env_id = env_info["env_id"]
ori_env_kwargs = env_info["env_kwargs"]
env = gym.make(env_id, **ori_env_kwargs)
success_count = 0
from collections import deque
current_obs = deque(maxlen=16)
#env.reset(seed=_seed,options=dict(reconfigure=True,hero_object_id=ObjectId.MONSTER_CAN,scene_id=SceneId.KITCHEN_COUNTER)) # reset with a seed for determinism
for ix in range(len(json_data["episodes"])):
    episode = json_data["episodes"][ix] # picks the first
    env.reset(**episode["reset_kwargs"])# reset with a seed for determinism
        #pick_cube_env.reset(options=dict(reconfigure=True)) # reset with a seed for determinism
    for i in range(16):
        obs, _, _, _,_ = env.step(get_action())
        a,b,c = get_relevant_info(env)
        b = b.reshape(-1)
        current_obs.append(np.concatenate([b, [c]]))
    policy = LLDM_Policy(current_obs= torch.tensor(np.array(current_obs)).unsqueeze(0))
    for _ in range(50):
        a,b,c = get_relevant_info(env)
        action_chunk = policy.predict_action(a,b,c)
        print('yo',action_chunk.shape)
        for step in range(16-2):
            pred = action_chunk[step]
            relT = pred[:16].reshape(4,4)
            grip = pred[16]
            action = get_action(relT, grip)
            obs, _, _, _,info = env.step(action)
            b,c = get_real_time_updates(obs)
            b = b.reshape(-1)
            current_obs.append(np.concatenate([b, [c]]))
            policy.update_last_obs(torch.tensor(np.array(current_obs)).unsqueeze(0))
            env.render()
            time.sleep(0.1)
            # for i in range(32):
            #     action = get_action(eef_poses[i],2*(gripper_poss[i]-0.5))
            #     for ix in range(1):
            #         obs, _, _, _,info = pick_cube_env.step(action)
            #         pick_cube_env.render()
            #     a,b,c = get_relevant_info(obs)
            #     #print(f"After setting state {gripper_poss[i]*0.04} {pick_cube_env.agent.robot.qpos[0,7]}")
            #     state_buffer.set_state(b,a,c)
        
    #print(info['is_grasped'])
    if info['is_grasped']:
        print("Grasped")
        success_count += 1
    #print(f"Finished episode {imx}/50 | Success: {success_count}/{imx+1}")




