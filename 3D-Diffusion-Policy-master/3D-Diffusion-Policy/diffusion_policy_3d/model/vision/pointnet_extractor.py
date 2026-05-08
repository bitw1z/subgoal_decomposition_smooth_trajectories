import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import einops
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule, RotaryPositionEncoding3D

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.BatchNorm1d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# Xinyu's version of DP3 
'''
class Act3dEncoder(nn.Module):
    def __init__(self, 
                 in_channels=6, 
                 encoder_output_dim=60, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 goal_mode='None',
                 mode=None,
                 use_mlp=False,
                 self_attention=False,
                 use_attn_for_point_features=False,
                 pointcloud_backbone='mlp',
                 use_lightweight_unet=False,
                 final_attention=False,
                 attention_num_heads=3,
                 attention_num_layers=2,
                 use_repr_10d=False,
                 eef_points=4,
                 embedding_type="shared",
                 **kwargs
                 ):
        super(Act3dEncoder, self).__init__()
        hidden_layer_dim = encoder_output_dim
        self.goal_mode = goal_mode
        self.eef_points = eef_points
        self.embedding_type = embedding_type

        vision_encoder = nn.Sequential(       # A 3-layer MLP that processes each point cloud
            nn.Linear(in_channels, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, encoder_output_dim)
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)     # replaces BatchNorm with GroupNorm

        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        attn_layers = replace_bn_with_gn(attn_layers)

        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })

        position_embedding_mlp = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),       # 9 = in_channels + 3
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        
        self.nets['gripper_pcd_position_embedding_mlp'] = position_embedding_mlp
        
        if self.embedding_type == "shared":
            self.nets['embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
        elif self.embedding_type == "separate":
            self.nets['embed'] = nn.Embedding(eef_points, encoder_output_dim // 3 * 2)
        else:
            raise NotImplementedError

        goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
        self.nets['goal_attn_layers'] = goal_attn_layers
        self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)

        if self.embedding_type == "shared":
            self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
        elif self.embedding_type == "separate":
            self.nets['goal_embed'] = nn.Embedding(eef_points, encoder_output_dim // 3 * 2)

    def forward(self, x):
        # x shape: [B, hor, N, input_dim]

        if self.goal_mode == 'None':
            x[..., NUM_SCENE_PCD+NUM_HAND_PCD:, :] = 0

        # scene point cloud
        if self.eef_points == 4:
            chosen_four_point_idx = torch.tensor([16, 40, 64, 88])
        elif self.eef_points == 12:
            chosen_four_point_idx = torch.tensor([4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92])
        else:
            raise ValueError("Chosen eef_points have not been implemented.")
        
        point_cloud = x[..., :NUM_SCENE_PCD, :]

        B, N, C = point_cloud.shape
        point_cloud_flatten = point_cloud.reshape(-1, C)
        point_cloud_features_flatten = self.nets['vision_encoder'](point_cloud_flatten)
        point_cloud_features = point_cloud_features_flatten.reshape(B, N, -1)
        point_cloud_features = einops.rearrange(point_cloud_features, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim
        point_cloud_rel_pos_embedding = self.nets['relative_pe_layer'](point_cloud)  # B N encoder_output_dim

        # attention between gripper pcd and scene pcd
        gripper_pcd = x[..., NUM_SCENE_PCD + chosen_four_point_idx, :]
        gripper_pcd_rel_pos_embedding = self.nets['relative_pe_layer'](gripper_pcd)  # B num_gripper_points encoder_output_dim
        
        if self.embedding_type == "shared":
            gripper_pcd_features = self.nets['embed'].weight.unsqueeze(0).repeat(self.eef_points, B, 1)  # num_gripper_points B encoder_output_dim
        elif self.embedding_type == "separate":
            gripper_pcd_features = self.nets['embed'].weight.unsqueeze(1).repeat(1, B, 1) # num_gripper_points B encoder_output_dim
        

        displacement_to_goal = x[..., NUM_SCENE_PCD + NUM_HAND_PCD + chosen_four_point_idx, :3] - x[..., NUM_SCENE_PCD + chosen_four_point_idx, :3]
        input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal], dim=-1)  # B num_gripper_points (in_channels+3)
        gripper_pcd_position_embedding = self.nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
        gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

        # print(gripper_pcd_features.shape)
        # print(gripper_pcd_position_embedding.shape)
        # breakpoint()

        gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

        attn_output = self.nets['attn_layers'](
            query=gripper_pcd_features, value=point_cloud_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]  # N B encoder_output_dim

        # goal gripper
        goal_gripper_pcd = x[..., NUM_SCENE_PCD + NUM_HAND_PCD + chosen_four_point_idx, :]
        goal_gripper_pcd_rel_pos_embedding = self.nets['relative_pe_layer'](goal_gripper_pcd)
        
        if self.embedding_type == "shared":
            goal_gripper_pcd_features = self.nets['goal_embed'].weight.unsqueeze(0).repeat(self.eef_points, B, 1)
        elif self.embedding_type == "separate":
            goal_gripper_pcd_features = self.nets['goal_embed'].weight.unsqueeze(1).repeat(1, B, 1) # num_gripper_points B encoder_output_dim
        

        displacement_to_goal = goal_gripper_pcd[..., :3] - gripper_pcd[..., :3]
        input_to_position_embedding = torch.cat([goal_gripper_pcd, displacement_to_goal], dim=-1)
        goal_gripper_pcd_position_embedding = self.nets['goal_pcd_position_embedding_mlp'](input_to_position_embedding)
        goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

        goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)

        goal_attn_output = self.nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,
                    query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
                )[-1]

        obs_feature = torch.cat([attn_output, goal_attn_output], dim=-1)
        obs_feature = einops.rearrange(obs_feature, "N B encoder_output_dim -> B N encoder_output_dim")

        return obs_feature.flatten(start_dim=1)
'''

# Yufei's version in articubot
class Act3dEncoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 encoder_output_dim=256, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 goal_mode=None,
                 mode=None,
                 use_mlp=True,
                 self_attention=False,
                 pointcloud_backbone='mlp',
                 final_attention=False,
                 attention_num_heads=3,
                 attention_num_layers=2,
                 **kwargs
                 ):
        super(Act3dEncoder, self).__init__()
        
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.gripper_pcd_key = 'gripper_pcd'
        self.num_gripper_points = num_gripper_points
        self.encoder_output_dim = encoder_output_dim
        self.state_shape = observation_space[self.state_key]
        self.goal_mode = goal_mode
        self.use_mlp = use_mlp
        
        self.self_attention = self_attention
        self.final_attention = final_attention
        self.mode = mode
        if self.mode in ['keep_position_feature_in_attention_feature']:
            vision_output_dim = encoder_output_dim // 3 * 2
        else:
            vision_output_dim = encoder_output_dim
        
        vision_encoder = None

        self.pointcloud_backbone = pointcloud_backbone
        if self.use_mlp:
            self.pointcloud_backbone = 'mlp'
        cprint("Using pointcloud backbone: " + self.pointcloud_backbone, 'green')

        if self.pointcloud_backbone == 'mlp':
            hidden_layer_dim = encoder_output_dim
            vision_encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, encoder_output_dim)
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            cprint(f"Unknown pointcloud backbone {self.pointcloud_backbone}", 'red')
            
        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        attn_layers = replace_bn_with_gn(attn_layers)
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })
        
        if self.self_attention:
            self.nets['self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self.nets['self_attn_layers'] = replace_bn_with_gn(self.nets['self_attn_layers'])
        
        input_dim = 3
        if self.goal_mode is not None:
            input_dim += 3
        if self.mode == "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object":
            input_dim += 3
        position_embedding_mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        object_pcd_position_embedding_mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        self.nets['object_pcd_position_embedding_mlp'] = object_pcd_position_embedding_mlp
        self.nets['gripper_pcd_position_embedding_mlp'] = position_embedding_mlp
        self.nets['embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

        # NOTE: 
        # here is how the low-level policy works:
        # cross attention between current gripper and object -> vec1
        # cross attention between current gripper and target gripper -> vec2
        # concatenate vectors -> diffusion -> output
        
        assert self.goal_mode == 'cross_attention_to_goal', "goal_mode must be 'cross_attention_to_goal' for Act3dEncoder, got {}".format(self.goal_mode)
        goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
        self.nets['goal_attn_layers'] = goal_attn_layers
        
        '''
        if self.mode in ['keep_position_feature_in_attention_feature', "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"]:
            self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
            self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
        '''

        if self.goal_mode == 'cross_attention_to_goal':
            self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
            self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

        if self.self_attention: ### add more self attention layers
            self.nets['goal_self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)   # [Debug] make it deeper
            self.nets['goal_self_attn_layers'] = replace_bn_with_gn(self.nets['goal_self_attn_layers'])

        if self.final_attention: ### add more self attention layers
            self.nets['final_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self.nets['final_attn_layers'] = replace_bn_with_gn(self.nets['final_attn_layers'])
            self.nets['final_slef_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self.nets['final_slef_attn_layers'] = replace_bn_with_gn(self.nets['final_slef_attn_layers'])
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = encoder_output_dim * self.num_gripper_points
        self.n_output_channels += output_dim
        if self.goal_mode == 'cross_attention_to_goal' and not self.final_attention: 
            self.n_output_channels += encoder_output_dim * self.num_gripper_points
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observation: Dict, return_full=False) -> torch.Tensor:
        # NOTE: the things passed in is already flattend from B, T, ... -> B*T, ...
        nets = self.nets
        
        agent_pos = observation[self.state_key]
        B = agent_pos.shape[0] #  B = batch_size * obs_horizon

        pcd = observation[self.point_cloud_key]
        B, N, C = pcd.shape
        pcd_obs_flatten = pcd.reshape(-1, C)
        pcd_features_flatten = nets['vision_encoder'](pcd_obs_flatten)
        pcd_features = pcd_features_flatten.reshape(B, N, -1) # shape B N encoder_output_dim
        pcd_features = einops.rearrange(pcd_features, "B N encoder_output_dim -> N B encoder_output_dim") # shape N B encoder_output_dim
        
        point_cloud = observation[self.point_cloud_key]
        point_cloud_rel_pos_embedding = nets['relative_pe_layer'](point_cloud) # shape B N encoder_output_dim
        num_gripper_points = observation['gripper_pcd'].shape[1] # gripper pcd is B num_gripper_points 3
        assert num_gripper_points == self.num_gripper_points, f"Expected {self.num_gripper_points} gripper points, got {num_gripper_points}"
        gripper_pcd = observation[self.gripper_pcd_key]
        gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](gripper_pcd) # shape B num_gripper_points encoder_output_dim
        gripper_pcd_features = nets['embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
        
        displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
        input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal], dim=-1)
        if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
            displacement_to_closest_object = observation['displacement_gripper_to_object']
            input_to_position_embedding = torch.cat([input_to_position_embedding, displacement_to_closest_object], dim=-1)
        input_to_position_embedding = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
        gripper_pcd_position_embedding = nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
        gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=num_gripper_points)
        gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

        self._pcd_features = pcd_features
        self._point_cloud = point_cloud
        attn_output = nets['attn_layers'](
            query=gripper_pcd_features, value=pcd_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]
        
        if not self.self_attention:
            pcd_features = einops.rearrange(
                attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # shape B (num_gripper_points * encoder_output_dim)
        else:
            self_attn_output = nets['self_attn_layers'](
                query=attn_output, value=attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]
            pcd_features = einops.rearrange(
                self_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
            
        state_feat = self.state_mlp(agent_pos)  # B * 64
        obs_features = torch.cat([pcd_features, state_feat], dim=-1)

        goal_gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](observation['goal_gripper_pcd']) # shape B num_gripper_points encoder_output_dim
        goal_gripper_pcd_features = nets['goal_embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
        displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
        input_to_position_embedding = torch.cat([observation['goal_gripper_pcd'], displacement_to_goal], dim=-1)
        if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
            displacement_to_closest_object = observation['displacement_gripper_to_object']
            input_to_position_embedding = torch.cat([input_to_position_embedding, displacement_to_closest_object], dim=-1)
        goal_gripper_pcd_position = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
        goal_gripper_pcd_position_embedding = nets['goal_pcd_position_embedding_mlp'](goal_gripper_pcd_position)
        goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=self.num_gripper_points)
        goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)
                
        goal_attn_output = nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
        )[-1]
        
        if self.self_attention:
            goal_attn_output = nets['goal_self_attn_layers'](query=goal_attn_output, value=goal_attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]

        
        if self.final_attention:
            final_attn_output = nets['final_attn_layers'](query=attn_output, value=goal_attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
            )[-1]
            final_attn_output = nets['final_slef_attn_layers'](query=final_attn_output, value=final_attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]
            obs_features = einops.rearrange(
                final_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
                
            obs_features = torch.cat([obs_features, state_feat], dim=-1)     
        else:
            goal_features = einops.rearrange(
                goal_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)

            obs_features = torch.cat([obs_features, goal_features], dim=-1)    
            
        return obs_features
    
    def output_shape(self):
        return self.n_output_channels
    

    def get_pcd_features(self):
        return self._point_cloud, self._pcd_features

class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class DP3Encoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 goal_mode='None',
                 eef_points=4,
                 embedding_type='shared',
                 ):
        super().__init__()

        # ---- keys ----
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.goal_gripper_key = 'goal_gripper_pcd'

        # ---- store flags ----
        self.pointnet_type = pointnet_type
        self.goal_mode = goal_mode
        self.eef_points = eef_points
        self.embedding_type = embedding_type
        self.use_pc_color = use_pc_color

        # ---- observation shapes ----
        self.use_imagined_robot = self.imagination_key in observation_space
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] use imagined robot: {self.use_imagined_robot}", "yellow")

        # ======================================================
        # 1) Build pointcloud extractor and FIX its output dim
        # ======================================================
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)

            self.extractor_out_dim = out_channel

        elif pointnet_type == "act3d":
            act3d_observation_space = {
                self.state_key: self.state_shape
            }
            self.extractor = Act3dEncoder(
                goal_mode=self.goal_mode,
                observation_space=act3d_observation_space,
                **pointcloud_encoder_cfg
            )

            # Act3D already includes state internally
            self.extractor_out_dim = self.extractor.output_shape()

        else:
            raise NotImplementedError(f"Unsupported pointnet_type: {pointnet_type}")

        # ======================================================
        # 2) Build state MLP (same as original DP3)
        # ======================================================
        if self.pointnet_type == "act3d":
            self.state_out_dim = 0
            self.state_mlp = None
        else:
            if len(state_mlp_size) == 0:
                raise RuntimeError("state_mlp_size cannot be empty")

            if len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]

            self.state_out_dim = state_mlp_size[-1]
            self.state_mlp = nn.Sequential(
                *create_mlp(
                    self.state_shape[0],
                    self.state_out_dim,
                    net_arch,
                    state_mlp_activation_fn
                )
            )
        
        # ======================================================
        # 3) FINAL encoder output dimension (single source of truth)
        # ======================================================
        self.n_output_channels = self.extractor_out_dim + self.state_out_dim
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

        # -----------------------
        # Language FiLM
        # -----------------------
        self.lang_key = "lang"
        self.lang_dim = 512
        self.use_film = True   # make it configurable later if you want

        if self.use_film:
            # condition ONLY the pointcloud feature path first (stable)
            self.film_pn = nn.Linear(self.lang_dim, 2 * self.extractor_out_dim)

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, observations: Dict) -> torch.Tensor:

        # -------- point cloud processing --------
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, \
            cprint(f"point cloud shape {points.shape} invalid", "red")

        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]
            points = torch.cat([points, img_points], dim=1)

        if self.pointnet_type == "pointnet":
            pn_feat = self.extractor(points)

        elif self.pointnet_type == "act3d":
            obs_dict = {
                "agent_pos": observations["agent_pos"],
                "point_cloud": observations["point_cloud"],
                "gripper_pcd": observations["gripper_pcd"],
                "goal_gripper_pcd": observations["goal_gripper_pcd"],
            }

            pn_feat = self.extractor(obs_dict)
        else:
            raise RuntimeError("Invalid pointnet_type")

        # -----------------------
        # FiLM conditioning
        # -----------------------
        if self.use_film:
            assert self.lang_key in observations, observations.keys()
            lang = observations[self.lang_key]  # (B,512) OR (B,1,512)
            if lang.dim() == 3:
                lang = lang[:, 0]               # (B,512)

            gb = self.film_pn(lang)             # (B, 2*C_pn)
            gamma, beta = gb.chunk(2, dim=-1)   # (B,C_pn), (B,C_pn)

            gamma = 1.0 + gamma                 # identity init effect
            pn_feat = gamma * pn_feat + beta

        if self.pointnet_type == "act3d":
            final_feat = pn_feat
        else:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)

        # -------- safety check --------
        if final_feat.shape[-1] != self.n_output_channels:
            raise RuntimeError(
                f"DP3Encoder output mismatch: got {final_feat.shape[-1]}, "
                f"expected {self.n_output_channels}. "
                f"(pn_feat={pn_feat.shape[-1]}, state_feat={state_feat.shape[-1]}, "
                f"extractor_out_dim={self.extractor_out_dim}, state_out_dim={self.state_out_dim})"
            )

        return final_feat

    def output_shape(self):
        return self.n_output_channels
