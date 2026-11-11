import numpy as np
import torch
import torch.nn as nn

from model_jta import (AuxilliaryEncoderCMT, AuxilliaryEncoderST,
                       LearnedTrajandIDEncoding,
                       create_model as create_model_baseline)

# Layout of the 47 slots per agent in the dataset:
# 0 = pelvis trajectory, 1 = 3D box, 2 = 2D box, 3..24 = 3D pose (22 joints), 25..46 = 2D pose (22 joints)
POSE3D_START = 3
POSE3D_END = 25


class LearnedPooledPoseEncoding(nn.Module):
    """Learned position embedding for the pooled pose tokens (one token per observed frame)."""
    def __init__(self, d_model, dropout=0.1, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x):
        # x: [B, in_F, N, d_model]
        seq_len = x.size(1)
        pos = self.learned_encoding(torch.arange(seq_len).to(self.device))  # [in_F, d_model]
        x = x + pos.unsqueeze(1).unsqueeze(0)                               # [1, in_F, 1, d_model]
        return self.dropout(x)


class TransMotionPooled(nn.Module):
    """Social-Transmotion with the 3D pose pooled into ONE token per frame.

    Local-former sequence per agent: 21 trajectory tokens + 9 pooled pose tokens = 30
    (the original uses 21 + 198 = 219 for traj + 3D pose).
    Boxes and 2D pose are not used in this version.
    """
    def __init__(self, nhid=128, nhead=4, dim_feedfwd=1024, nlayers_local=6, nlayers_global=3,
                 dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,
                 input_track_size=9, num_tokens=47, mask_ratio_traj=0.1, mask_ratio=0.1,
                 mask_ratio_modality=0.3, device='cuda:0'):
        super().__init__()
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.joints_pose = 22
        self.obs_and_pred = obs_and_pred
        self.in_F = input_track_size
        self.seq_len = obs_and_pred + input_track_size  # 21 + 9 = 30
        self.device = device

        self.mask_ratio_traj = mask_ratio_traj
        self.mask_ratio = mask_ratio
        self.mask_ratio_modality = mask_ratio_modality

        self.fc_in_traj = nn.Linear(2, nhid)
        self.fc_out_traj = nn.Linear(nhid, 2)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=obs_and_pred, device=device)

        # MLP over the 22 joints x 3 coordinates of one frame -> one token
        self.pose_pool = nn.Sequential(
            nn.Linear(self.joints_pose * 3, 2 * nhid),
            nn.ReLU(),
            nn.Linear(2 * nhid, nhid),
        )
        self.pose_pos_encoder = LearnedPooledPoseEncoding(nhid, dropout, seq_len=input_track_size, device=device)

        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead,
                                                         dim_feedforward=dim_feedfwd,
                                                         dropout=dropout, activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead,
                                                          dim_feedforward=dim_feedfwd,
                                                          dropout=dropout, activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)

    def forward(self, tgt, padding_mask, metamask=None):
        B, in_F, NJ, K = tgt.shape
        F = self.obs_and_pred
        J = self.token_num
        out_F = F - in_F
        N = NJ // J

        # pad the future frames with the last observed frame (same as the original model)
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:, i_idx]
        tgt = tgt.reshape(B, F, N, J, K)

        # masking is only active while training
        p_traj = self.mask_ratio_traj if self.training else 0.0
        p_joint = self.mask_ratio if self.training else 0.0
        p_modality = self.mask_ratio_modality if self.training else 0.0

        # ---- trajectory tokens: [B, F, N, d] ----
        tgt_traj = tgt[:, :, :, 0, :2].to(self.device)
        traj_keep = (torch.rand((B, F, N), device=self.device) > p_traj).float().unsqueeze(3)
        tgt_traj = tgt_traj * traj_keep
        tgt_traj = self.fc_in_traj(tgt_traj)
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)

        # ---- pooled 3D pose tokens: [B, in_F, N, d] ----
        tgt_pose = tgt[:, :in_F, :, POSE3D_START:POSE3D_END, :3].to(self.device)  # [B, in_F, N, 22, 3]
        modality_keep = (torch.rand((B, 1, N, 1, 1), device=self.device) > p_modality).float()
        joint_keep = (torch.rand((B, in_F, N, self.joints_pose, 1), device=self.device) > p_joint).float()
        tgt_pose = tgt_pose * modality_keep * joint_keep
        tgt_pose = tgt_pose.reshape(B, in_F, N, self.joints_pose * 3)             # [B, in_F, N, 66]
        tgt_pose = self.pose_pool(tgt_pose)                                       # [B, in_F, N, d]
        tgt_pose = self.pose_pos_encoder(tgt_pose)

        # ---- padding masks ----
        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len, dim=1)

        # ---- local former input: [30, B*N, d] ----
        tgt_traj = torch.transpose(tgt_traj, 0, 1).reshape(F, -1, self.nhid)
        tgt_pose = torch.transpose(tgt_pose, 0, 1).reshape(in_F, -1, self.nhid)
        tgt = torch.cat((tgt_traj, tgt_pose), 0)

        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        out_local = out_local * self.output_scale + tgt  # local residual

        # keep only the trajectory tokens and let agents interact in the global former
        out_local = out_local[:F].reshape(F, B, N, self.nhid).permute(2, 0, 1, 3).reshape(-1, B, self.nhid)
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)
        out_global = out_global * self.output_scale + out_local  # global residual

        out_primary = out_global.reshape(N, F, out_global.size(1), self.nhid)[0]
        out_primary = self.fc_out_traj(out_primary)
        out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
        return out


def create_model(config, logger):
    m = config["MODEL"]

    # keeps the original model available, so this file can also evaluate baseline checkpoints
    if m["type"] == "transmotion":
        return create_model_baseline(config, logger)

    if m["type"] == "transmotion_pooled":
        in_F = config["TRAIN"]["input_track_size"]
        out_F = config["TRAIN"]["output_track_size"]
        expected_len = (in_F + out_F) + in_F
        assert m["seq_len"] == expected_len, \
            f"MODEL.seq_len must be {expected_len} (= {in_F + out_F} trajectory + {in_F} pooled pose tokens)"

        logger.info("Creating pooled-joint-token model.")
        model = TransMotionPooled(
            nhid=m["dim_hidden"],
            nhead=m["num_heads"],
            dim_feedfwd=m["dim_feedforward"],
            nlayers_local=m["num_layers_local"],
            nlayers_global=m["num_layers_global"],
            output_scale=m["output_scale"],
            obs_and_pred=in_F + out_F,
            input_track_size=in_F,
            num_tokens=m["token_num"],
            mask_ratio_traj=m.get("mask_ratio_traj", 0.1),
            mask_ratio=m.get("mask_ratio", 0.1),
            mask_ratio_modality=m.get("mask_ratio_modality", 0.3),
            device=config["DEVICE"],
        ).to(config["DEVICE"]).float()
        return model

    raise ValueError(f"Model type '{m['type']}' not found")