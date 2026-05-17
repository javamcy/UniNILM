"""
相对于v2版本修改了NILMDataset类，BiMamba类(即是我们的UniNILM方法)，训练函数的数据加载部分和train_model中的数据迭代
目的是引入真正的Day-type gating门控机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import os
import json
import argparse
import psutil
import math
import gc
from torch.utils.data import Dataset, DataLoader

try:
    import mamba_ssm
    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False
    print("[WARN] mamba_ssm not installed, Mamba2NILM will not be available")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r'E:\data\NILM\Ampsd2\ampds2_aligned.csv' if os.name == 'nt' else '/mnt/e/data/NILM/Ampsd2/ampds2_aligned.csv'
RESULT_DIR = os.path.join(SCRIPT_DIR, 'trained_models')

APPLIANCE_NAMES = ['BME', 'DWE', 'FGE', 'TVE', 'RSE', 'DNE', 'HPE', 'CWE']
N_APPLIANCES = 8

TRAIN_SIZE = 1440 * 7 * 8  # 80640 (8周)
VAL_SIZE = 1440 * 7 * 8    # 80640 (8周)
SEQ_LENGTH = 480


class NILMDataset(Dataset):
    def __init__(self, data_path, seq_length=480, split='train', 
                 train_size=TRAIN_SIZE, val_size=VAL_SIZE, appliance_idx=None,
                 sp0_offset=0, appliance_columns=None, use_day_type=True):
        """
        Args:
            use_day_type: 是否启用日期类型特征（工作日/周末区分）
        """
        self.seq_length = seq_length
        self.split = split
        self.appliance_idx = appliance_idx
        self.use_day_type = use_day_type
        
        if appliance_columns is None:
            appliance_columns = APPLIANCE_NAMES
        self.appliance_columns = appliance_columns
        
        print(f"Loading data from {data_path}...", flush=True)
        df = pd.read_csv(data_path)
        
        self.timestamps = df['unix_ts'].values.astype(np.int64)
        self.appliance_data_raw = df[self.appliance_columns].values.astype(np.float32)
        self.total_power_raw = df['SP'].values.astype(np.float32)
        
        # ========== 日期类型特征提取 ==========
        if self.use_day_type:
            datetimes = pd.to_datetime(self.timestamps, unit='s')
            # 星期几 (0=Monday, 6=Sunday) 归一化到 [0,1]
            self.weekday_norm = datetimes.dayofweek.values.astype(np.float32) / 6.0
            # 是否周末 (周六=5, 周日=6)
            self.is_weekend = (datetimes.dayofweek >= 5).astype(np.float32)
            # 是否工作日
            self.is_weekday = (datetimes.dayofweek < 5).astype(np.float32)
            # 小时 (0-23) 归一化到 [0,1]
            self.hour_norm = datetimes.hour.values.astype(np.float32) / 23.0
            # 分钟归一化 (0-59) -> [0,1]
            self.minute_norm = datetimes.minute.values.astype(np.float32) / 59.0
            # 组合特征: [weekday_norm, is_weekend, is_weekday, hour_norm, minute_norm]
            self.day_type_features = np.stack([
                self.weekday_norm,
                self.is_weekend,
                self.is_weekday,
                self.hour_norm,
                self.minute_norm
            ], axis=-1)
            print(f"  Day-type features (5-dim): {self.day_type_features.shape}", flush=True)
        
        if sp0_offset > 0:
            self.total_power_raw = self.total_power_raw[sp0_offset:]
            self.appliance_data_raw = self.appliance_data_raw[sp0_offset:]
            self.timestamps = self.timestamps[sp0_offset:]
            if self.use_day_type:
                self.day_type_features = self.day_type_features[sp0_offset:]
            print(f"  Skipped first {sp0_offset} rows with SP=0, remaining: {len(self.total_power_raw)}", flush=True)
        
        # 归一化
        train_mean = self.total_power_raw[:train_size].mean()
        train_std = self.total_power_raw[:train_size].std() + 1e-8
        self.total_power_norm = (self.total_power_raw - train_mean) / train_std
        
        n_samples = len(self.total_power_raw) - seq_length
        train_end = train_size
        val_end = train_size + val_size
        
        if split == 'train':
            self.indices = np.arange(0, train_end - seq_length)
        elif split == 'val':
            self.indices = np.arange(train_end, val_end - seq_length)
        else:
            self.indices = np.arange(val_end, n_samples)
        
        print(f"{split} samples: {len(self.indices)}", flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seq_length
        
        total_norm = self.total_power_norm[start:end].reshape(-1, 1)
        total_raw = self.total_power_raw[start:end]
        
        if self.appliance_idx is not None:
            target = self.appliance_data_raw[start:end, self.appliance_idx:self.appliance_idx+1]
        else:
            target = self.appliance_data_raw[start:end]
        
        if self.use_day_type:
            day_type = self.day_type_features[start:end]  # [seq_len, 5]
            return (
                torch.FloatTensor(total_norm),
                torch.FloatTensor(total_raw),
                torch.FloatTensor(target),
                torch.LongTensor([start]),
                torch.FloatTensor(day_type)
            )
        else:
            return (
                torch.FloatTensor(total_norm),
                torch.FloatTensor(total_raw),
                torch.FloatTensor(target),
                torch.LongTensor([start])
            )


class Seq2PointSeq2Seq(nn.Module):
    """真正的Seq2Point：输出序列中心点的预测"""
    def __init__(self, input_dim=1, n_appliances=1, seq_length=480):
        super().__init__()
        self.seq_length = seq_length
        self.center_idx = seq_length // 2
        
        self.conv1 = nn.Conv1d(input_dim, 30, kernel_size=10, padding=5)
        self.conv2 = nn.Conv1d(30, 30, kernel_size=8, padding=4)
        self.conv3 = nn.Conv1d(30, 40, kernel_size=6, padding=3)
        self.conv4 = nn.Conv1d(40, 50, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(50, 50, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten_dim = 50 * (seq_length // 8)
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, n_appliances)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, total_raw=None, sample_positions=None):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x)); x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        point_out = self.fc3(x)
        return point_out, None, None, None


class Seq2Seq(nn.Module):
    def __init__(self, input_dim=1, n_appliances=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_appliances)
        
    def forward(self, x, total_raw=None, sample_positions=None):
        enc_out, _ = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        return self.fc(dec_out), None, None, None


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PreFeatureExtractor(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.branch3 = ConvBlock(in_ch, out_ch // 4, kernel_size=3, padding=1)
        self.branch7 = ConvBlock(in_ch, out_ch // 4, kernel_size=7, padding=3)
        self.branch15 = ConvBlock(in_ch, out_ch // 4, kernel_size=15, padding=7)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(in_ch, out_ch // 4, 1),
            nn.BatchNorm1d(out_ch // 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.float()
        b3 = self.branch3(x)
        b7 = self.branch7(x)
        b15 = self.branch15(x)
        bp = self.branch_pool(x)
        out = torch.cat([b3, b7, b15, bp], dim=1)
        return self.fusion(out)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super().__init__()
        bc = base_ch
        
        self.enc1 = nn.Sequential(nn.Conv1d(in_ch, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(bc, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(bc*2, bc*4, 3, padding=1), nn.BatchNorm1d(bc*4), nn.ReLU())
        
        self.pool = nn.MaxPool1d(2)
        
        self.up2 = nn.ConvTranspose1d(bc*4, bc*2, 2, stride=2)
        self.up1 = nn.ConvTranspose1d(bc*2, bc, 2, stride=2)
        
        self.dec2 = nn.Sequential(nn.Conv1d(bc*4, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.PReLU())
        self.dec1 = nn.Sequential(nn.Conv1d(bc*2, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.PReLU())
        
        self.out_conv = nn.Conv1d(bc, out_ch, 1)
    
    def forward(self, x, return_enc_features=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.out_conv(d1)
        
        if return_enc_features:
            return out, [e1, e2, e3]
        return out


class SimpleUNetWithSkip(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super().__init__()
        bc = base_ch
        
        self.enc1 = nn.Sequential(nn.Conv1d(in_ch, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(bc, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(bc*2, bc*4, 3, padding=1), nn.BatchNorm1d(bc*4), nn.ReLU())
        
        self.pool = nn.MaxPool1d(2)
        
        self.up2 = nn.ConvTranspose1d(bc*4, bc*2, 2, stride=2)
        self.up1 = nn.ConvTranspose1d(bc*2, bc, 2, stride=2)
        
        self.dec2 = nn.Sequential(nn.Conv1d(bc*4, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.PReLU())
        self.dec1 = nn.Sequential(nn.Conv1d(bc*2, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.PReLU())
        
        self.out_conv = nn.Conv1d(bc, out_ch, 1)
    
    def forward(self, x, skip_features=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        if skip_features is not None:
            skip_e1, skip_e2, skip_e3 = skip_features
            e1 = e1 + skip_e1
            e2 = e2 + skip_e2
            e3 = e3 + skip_e3
        
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.out_conv(d1)
        return out


class DCTAttention(nn.Module):
    def __init__(self, channels, seq_length):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        
        self.freq_attn = nn.Sequential(
            nn.Linear(channels * seq_length, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels),
            nn.Sigmoid()
        )
    
    def dct(self, x, norm=None):
        x = x.float()
        N = x.size(-1)
        k = torch.arange(N, dtype=x.dtype, device=x.device).unsqueeze(0)
        n = torch.arange(N, dtype=x.dtype, device=x.device).unsqueeze(1)
        dct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        if norm == 'ortho':
            dct_matrix[0, :] *= 1 / math.sqrt(2 * N)
            dct_matrix[1:, :] *= math.sqrt(2 / N)
        else:
            dct_matrix[0, :] *= 1 / math.sqrt(2)
            dct_matrix *= math.sqrt(2 / N)
        
        return torch.matmul(x, dct_matrix)
    
    def forward(self, x):
        x = x.float()
        dct_out = self.dct(x, norm='ortho')
        dct_flat = dct_out.view(x.size(0), -1)
        af = self.freq_attn(dct_flat).unsqueeze(-1)
        return af


class DualDomainAttention(nn.Module):
    def __init__(self, channels, seq_length):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        self.freq_attn = DCTAttention(channels, seq_length)
    
    def forward(self, x_pre, x_u1):
        at = self.temporal_attn(x_u1)
        af = self.freq_attn(x_u1)
        
        out = x_pre * at * af
        return out


class DUNILM(nn.Module):
    def __init__(self, input_dim=1, n_appliances=1, base_channels=32, seq_length=480, N_segments=8):
        super().__init__()
        self.seq_length = seq_length
        self.N_segments = N_segments
        self.n_appliances = n_appliances
        bc = base_channels
        
        self.pre_extractor = PreFeatureExtractor(input_dim, bc)
        
        self.u1_encoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(bc, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.ReLU()),
            nn.Sequential(nn.Conv1d(bc, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.ReLU()),
            nn.Sequential(nn.Conv1d(bc*2, bc*4, 3, padding=1), nn.BatchNorm1d(bc*4), nn.ReLU())
        ])
        
        self.u1_decoder = nn.ModuleList([
            nn.ConvTranspose1d(bc*4, bc*2, 2, stride=2),
            nn.ConvTranspose1d(bc*2, bc, 2, stride=2)
        ])
        
        self.u1_dec_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(bc*4, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.PReLU()),
            nn.Sequential(nn.Conv1d(bc*2, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.PReLU())
        ])
        
        self.pool = nn.MaxPool1d(2)
        
        self.dual_attn = DualDomainAttention(bc, seq_length)
        
        self.u2_encoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(bc, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.ReLU()),
            nn.Sequential(nn.Conv1d(bc, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.ReLU()),
            nn.Sequential(nn.Conv1d(bc*2, bc*4, 3, padding=1), nn.BatchNorm1d(bc*4), nn.ReLU())
        ])
        
        self.u2_decoder = nn.ModuleList([
            nn.ConvTranspose1d(bc*4, bc*2, 2, stride=2),
            nn.ConvTranspose1d(bc*2, bc, 2, stride=2)
        ])
        
        self.u2_dec_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(bc*6, bc*2, 3, padding=1), nn.BatchNorm1d(bc*2), nn.PReLU()),
            nn.Sequential(nn.Conv1d(bc*3, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.PReLU())
        ])
        
        self.out_conv = nn.Conv1d(bc, n_appliances, 1)
        
        self.use_seq2seg = True
    
    def seq2seg_extraction(self, input_seq):
        B, C, L = input_seq.shape
        L_c = L // self.N_segments
        interval_len = (L - L_c * self.N_segments) // (self.N_segments + 1)
        
        segments = []
        for i in range(self.N_segments):
            start = interval_len + i * (L_c + interval_len)
            end = start + L_c
            segments.append(input_seq[:, :, start:end])
        
        return torch.cat(segments, dim=2)
    
    def forward(self, x, total_raw=None, sample_positions=None):
        x = x.float().transpose(1, 2)
        
        x_pre = self.pre_extractor(x)
        
        e1_1 = self.u1_encoder[0](x_pre)
        e1_2 = self.u1_encoder[1](self.pool(e1_1))
        e1_3 = self.u1_encoder[2](self.pool(e1_2))
        
        d1_2 = self.u1_dec_convs[0](torch.cat([self.u1_decoder[0](e1_3), e1_2], dim=1))
        d1_1 = self.u1_dec_convs[1](torch.cat([self.u1_decoder[1](d1_2), e1_1], dim=1))
        
        x_enhanced = self.dual_attn(x_pre, d1_1)
        
        e2_1 = self.u2_encoder[0](x_enhanced)
        e2_2 = self.u2_encoder[1](self.pool(e2_1))
        e2_3 = self.u2_encoder[2](self.pool(e2_2))
        
        e2_1 = e2_1 + e1_1
        e2_2 = e2_2 + e1_2
        e2_3 = e2_3 + e1_3
        
        d2_2 = self.u2_dec_convs[0](torch.cat([self.u2_decoder[0](e2_3), e2_2, e1_2], dim=1))
        d2_1 = self.u2_dec_convs[1](torch.cat([self.u2_decoder[1](d2_2), e2_1, e1_1], dim=1))
        
        output = self.out_conv(d2_1)
        
        if self.use_seq2seg and self.training:
            output = self.seq2seg_extraction(output)
        
        return output.transpose(1, 2), None, None, None


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class FFSTT(nn.Module):
    def __init__(self, input_dim=1, n_appliances=1, d_model=64, nhead=8, num_layers=2, seq_length=480):
        super().__init__()
        self.seq_length = seq_length
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_appliances)
        
    def forward(self, x, total_raw=None, sample_positions=None):
        enc_out = self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        output = self.output_proj(enc_out)
        return output, None, None, None


class AttentionNILM(nn.Module):
    def __init__(self, input_dim=1, n_appliances=1, d_model=64, nhead=4, seq_length=480):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.n_appliances = n_appliances
        
        branch_dim = d_model // 3
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_dim, branch_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_dim, branch_dim, 3, padding=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_dim, branch_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(branch_dim, branch_dim, 5, padding=2),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_dim, branch_dim, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(branch_dim, branch_dim, 7, padding=3),
            nn.ReLU()
        )
        
        self.input_proj = nn.Conv1d(input_dim, branch_dim, 1)
        
        self.fusion_proj = nn.Linear(branch_dim * 4, d_model)
        
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.output_proj = nn.Linear(d_model, n_appliances)
    
    def forward(self, x, total_raw=None, sample_positions=None):
        x_t = x.transpose(1, 2)
        
        b1 = self.branch1(x_t)
        b2 = self.branch2(x_t)
        b3 = self.branch3(x_t)
        b_input = self.input_proj(x_t)
        
        combined = torch.cat([b1, b2, b3, b_input], dim=1)
        combined = combined.transpose(1, 2)
        
        feat = self.fusion_proj(combined)
        
        attn_out, _ = self.mha(feat, feat, feat)
        feat = self.norm1(feat + attn_out)
        
        ffn_out = self.ffn(feat)
        feat = self.norm2(feat + ffn_out)
        
        output = self.output_proj(feat)
        
        return output, None, None, None


class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ESEBlock(nn.Module):
    """Energy-Aware Subsampling Embedding"""
    def __init__(self, input_dim, d_model, M=3):
        super().__init__()
        self.M = M
        self.d_model = d_model
        
        self.pool_layers = nn.ModuleList()
        self.token_embeds = nn.ModuleList()
        
        for m in range(M + 1):
            if m == 0:
                self.pool_layers.append(nn.Identity())
            else:
                self.pool_layers.append(nn.AvgPool1d(kernel_size=2**m, stride=2**m))
            
            self.token_embeds.append(nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1))
    
    def forward(self, x):
        P = []
        for m in range(self.M + 1):
            pooled = self.pool_layers[m](x)
            embedded = self.token_embeds[m](pooled)
            P.append(embedded)
        return P


class LDBlock(nn.Module):
    """Learnable Decomposition Block"""
    def __init__(self, d_model, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        
        U = torch.zeros(kernel_size)
        center = kernel_size // 2
        sigma = 1.0
        for a in range(kernel_size):
            U[a] = torch.exp(torch.tensor(-((a - center) ** 2) / (2 * sigma ** 2)))
        
        self.W = nn.Parameter(U.unsqueeze(0).unsqueeze(0))
        
    def forward(self, H):
        B, L, D = H.shape
        
        H_t = H.transpose(1, 2)
        
        W_normalized = F.softmax(self.W, dim=-1)
        W_expanded = W_normalized.expand(D, -1, -1)
        
        T = F.conv1d(H_t, W_expanded, padding=self.kernel_size // 2, groups=D)
        T = T.transpose(1, 2)
        
        S = H - T
        
        return S, T


class DTFBlock(nn.Module):
    """Dual-Path Temporal Fusion Block - 论文精确实现"""
    def __init__(self, d_model, n_heads=4, M=3, shift_len=64):
        super().__init__()
        self.M = M
        self.shift_len = shift_len
        
        self.channel_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.channel_fnn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.channel_norm1 = nn.LayerNorm(d_model)
        self.channel_norm2 = nn.LayerNorm(d_model)
        
        self.ar_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ar_fnn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ar_norm1 = nn.LayerNorm(d_model)
        self.ar_norm2 = nn.LayerNorm(d_model)
        
        self.bottom_up_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(M)
        ])
        
        self.top_down_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(M)
        ])
        
    def _channel_wise_attention(self, S_0):
        Ice = self.channel_norm1(S_0 + self.channel_attn(S_0, S_0, S_0)[0])
        S_ce_0 = self.channel_norm2(Ice + self.channel_fnn(Ice))
        return S_ce_0
    
    def _auto_regressive_attention(self, S_0):
        B, L, D = S_0.shape
        shift_len = min(self.shift_len, L)
        num_segments = L // shift_len + 1
        
        segments = []
        for p in range(num_segments):
            shift = (p * shift_len) % L
            shifted = torch.cat([S_0[:, shift:, :], S_0[:, :shift, :]], dim=1)
            segments.append(shifted)
        
        S_sa = torch.stack(segments, dim=2)
        S_sa, _ = S_sa.max(dim=2)
        
        Q = S_0
        K = V = S_sa
        
        Attn_out = self.ar_attn(Q, K, V)[0]
        
        Ice = self.ar_norm1(S_0 + Attn_out)
        S_ae_0 = self.ar_norm2(Ice + self.ar_fnn(Ice))
        return S_ae_0
    
    def forward(self, P_seasonal, P_trend):
        B = P_seasonal[0].size(0)
        L0 = P_seasonal[0].size(1)
        D = P_seasonal[0].size(2)
        
        S_ce_0 = self._channel_wise_attention(P_seasonal[0])
        S_ae_0 = self._auto_regressive_attention(P_seasonal[0])
        S_alpha_0 = S_ae_0 + S_ce_0
        
        S_alpha = [S_alpha_0]
        for m in range(1, self.M + 1):
            S_alpha.append(P_seasonal[m] * 2.0)
        
        S_beta = [S_alpha[0]]
        for m in range(1, self.M + 1):
            upsampled = F.interpolate(S_beta[m-1].transpose(1, 2), size=P_seasonal[m].size(1), mode='linear', align_corners=False)
            upsampled = upsampled.transpose(1, 2)
            mixed = self.bottom_up_mixers[m-1](upsampled)
            S_beta.append(S_alpha[m] + mixed)
        
        T_beta = [None] * (self.M + 1)
        T_beta[self.M] = P_trend[self.M]
        for m in range(self.M - 1, -1, -1):
            downsampled = F.interpolate(T_beta[m+1].transpose(1, 2), size=P_trend[m].size(1), mode='linear', align_corners=False)
            downsampled = downsampled.transpose(1, 2)
            mixed = self.top_down_mixers[m](downsampled)
            T_beta[m] = P_trend[m] + mixed
        
        return S_beta, T_beta


class RPDBlock(nn.Module):
    """Residual Power Decoupling Block - 论文精确实现"""
    def __init__(self, d_model, n_heads=4, M=3, N_R=2, shift_len=64):
        super().__init__()
        self.N_R = N_R
        self.M = M
        
        self.ld_blocks = nn.ModuleList([
            LDBlock(d_model, kernel_size=25) for _ in range(N_R)
        ])
        
        self.dtf_blocks = nn.ModuleList([
            DTFBlock(d_model, n_heads, M, shift_len) for _ in range(N_R)
        ])
        
    def forward(self, P):
        P_seasonal = [p.transpose(1, 2) for p in P]
        P_trend = [p.transpose(1, 2) for p in P]
        
        for i in range(self.N_R):
            new_seasonal = []
            new_trend = []
            
            for m in range(len(P_seasonal)):
                Y_fused = P_seasonal[m] + P_trend[m]
                S, T = self.ld_blocks[i](Y_fused)
                new_seasonal.append(S)
                new_trend.append(T)
            
            S_beta, T_beta = self.dtf_blocks[i](new_seasonal, new_trend)
            
            for m in range(len(P_seasonal)):
                P_seasonal[m] = P_seasonal[m] + S_beta[m]
                P_trend[m] = P_trend[m] + T_beta[m]
        
        return P_seasonal, P_trend


class AutoConLoss(nn.Module):
    """Autocorrelation Contrastive Learning Loss - 论文精确实现"""
    def __init__(self, d_model, gamma=0.07):
        super().__init__()
        self.gamma = gamma
        
        self.dc_block = nn.Sequential(
            DilatedConvBlock(d_model, d_model, dilation=1),
            DilatedConvBlock(d_model, d_model, dilation=2),
            DilatedConvBlock(d_model, d_model, dilation=4),
            DilatedConvBlock(d_model, d_model, dilation=8),
            DilatedConvBlock(d_model, d_model, dilation=16),
            nn.Conv1d(d_model, 256, kernel_size=1),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def compute_acf(self, x):
        """计算自相关函数 Γ[h] - 使用FFT加速"""
        B, L = x.shape
        
        x_mean = x.mean(dim=1, keepdim=True)
        delta = x - x_mean
        
        delta_padded = F.pad(delta, (0, L), mode='constant', value=0)
        delta_fft = torch.fft.rfft(delta_padded, dim=1)
        power_spectrum = delta_fft * torch.conj(delta_fft)
        auto_corr_full = torch.fft.irfft(power_spectrum, dim=1)
        auto_corr = auto_corr_full[:, :L]
        
        auto_corr = auto_corr / (auto_corr[:, 0:1] + 1e-8)
        
        return auto_corr
    
    def forward(self, T_0, total_raw=None, sample_positions=None):
        """
        T_0: (B, L_0, D) 最细尺度趋势序列
        total_raw: (B, L_0) 当前batch的总功率序列（用于计算自相关）
        sample_positions: (B,) 每个样本在总序列中的起始位置
        """
        B, L_0, D = T_0.shape
        
        T_0_t = T_0.transpose(1, 2)
        V_di = self.dc_block(T_0_t)
        
        l_max = V_di.abs().argmax(dim=2)
        V_gl = torch.gather(V_di, 2, l_max.unsqueeze(2)).squeeze(2)
        
        V_gl_norm = F.normalize(V_gl, dim=1)
        Sim = torch.mm(V_gl_norm, V_gl_norm.t())
        
        if total_raw is not None:
            acf = self.compute_acf(total_raw)
            
            if sample_positions is not None:
                pos_diff = torch.abs(sample_positions.unsqueeze(1) - sample_positions.unsqueeze(0))
                pos_diff = torch.clamp(pos_diff, max=L_0 - 1)
                
                batch_idx = torch.arange(B, device=T_0.device).unsqueeze(1).expand(B, B)
                Lambda = acf[batch_idx.flatten(), pos_diff.flatten()].view(B, B)
            else:
                idx_diff = torch.abs(torch.arange(B, device=T_0.device).unsqueeze(1) - 
                                    torch.arange(B, device=T_0.device).unsqueeze(0))
                idx_diff = torch.clamp(idx_diff, max=L_0 - 1)
                
                batch_idx = torch.arange(B, device=T_0.device).unsqueeze(1).expand(B, B)
                Lambda = acf[batch_idx.flatten(), idx_diff.flatten()].view(B, B)
            
            Lambda.fill_diagonal_(0)
        else:
            Lambda = torch.ones(B, B, device=T_0.device) * 0.5
            Lambda.fill_diagonal_(0)
        
        eta = torch.exp(Sim / self.gamma)
        
        mask = (Lambda.unsqueeze(1) <= Lambda.unsqueeze(2)).float()
        mask = mask * (1 - torch.eye(B, device=T_0.device).unsqueeze(1))
        
        denominator = (eta.unsqueeze(1) * mask).sum(dim=2)
        
        loss_matrix = -Lambda * torch.log(eta / (denominator + 1e-8) + 1e-8)
        loss = loss_matrix.sum() / (B * (B - 1))
        
        return loss


class CLMDTFN(nn.Module):
    def __init__(self, input_dim=1, n_appliances=8, d_model=64, use_contrastive=True, M=3, N_R=2):
        super().__init__()
        self.n_appliances = n_appliances
        self.use_contrastive = use_contrastive
        self.M = M
        
        self.ese = ESEBlock(input_dim, d_model, M)
        
        self.rpd = RPDBlock(d_model, n_heads=4, M=M, N_R=N_R)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_appliances)
        )
        
        if use_contrastive:
            self.autocon = AutoConLoss(d_model)
    
    def forward(self, x, total_raw=None, sample_positions=None):
        x_t = x.transpose(1, 2)
        
        P = self.ese(x_t)
        
        P_seasonal, P_trend = self.rpd(P)
        
        T_0 = P_trend[0]
        fused = P_seasonal[0] + P_trend[0]
        
        output = self.output_head(fused)
        
        if self.use_contrastive:
            contrastive_loss = self.autocon(T_0, total_raw, sample_positions)
            return output, None, None, contrastive_loss
        
        return output, None, None, None


class SEFFN(nn.Module):
    """Squeeze-and-Excitation Position-Wise Feed-Forward Network - 论文图3/4精确实现"""
    def __init__(self, d_model, d_ff=None, reduction_ratio=16, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(d_model, max(d_model // reduction_ratio, 1), bias=False),
            nn.ReLU(),
            nn.Linear(max(d_model // reduction_ratio, 1), d_model, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ffn_out = self.ffn(x)
        x_t = x.transpose(1, 2)
        squeezed = self.squeeze(x_t).squeeze(-1)
        channel_weights = self.excitation(squeezed).unsqueeze(1)
        out = self.dropout(ffn_out * channel_weights)
        return out


class Mamba2NILM(nn.Module):
    """Mamba2NILM: Hybrid Mamba2 + SEFFN for NILM - 论文图1/2/3精确实现"""
    def __init__(self, input_dim=1, n_appliances=8, d_model=240, n_layers=4,
                 reduction_ratio=16, dropout=0.1):
        super().__init__()
        self.n_appliances = n_appliances
        self.d_model = d_model
        self.tau = 0.1

        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout=dropout)

        self.mamba_layers = nn.ModuleList()
        self.seffn_layers = nn.ModuleList()
        self.norms_mamba = nn.ModuleList()
        self.norms_seffn = nn.ModuleList()

        for _ in range(n_layers):
            self.mamba_layers.append(
                mamba_ssm.Mamba(d_model, d_state=16, d_conv=4, expand=2)
            )
            self.seffn_layers.append(
                SEFFN(d_model, reduction_ratio=reduction_ratio, dropout=dropout)
            )
            self.norms_mamba.append(nn.LayerNorm(d_model))
            self.norms_seffn.append(nn.LayerNorm(d_model))

        self.decoder = nn.Linear(d_model, n_appliances)
        self.output_scale = nn.Parameter(torch.ones(n_appliances) * 10.0)
        self.output_bias = nn.Parameter(torch.zeros(n_appliances))

        self.use_mamba2nilm_loss = True

        self._init_weights()

    def _init_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, total_raw=None, sample_positions=None):
        x = self.enc_embedding(x)

        for mamba, seffn, norm_m, norm_s in zip(
            self.mamba_layers, self.seffn_layers,
            self.norms_mamba, self.norms_seffn
        ):
            residual = x
            x = mamba(x)
            x = norm_m(residual + x)

            residual = x
            x = seffn(x)
            x = norm_s(residual + x)

        output = self.decoder(x)
        output = output * self.output_scale + self.output_bias
        return output, None, None, None


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


if _HAS_MAMBA:
    class BidirectionalMambaBlock(nn.Module):
        def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.mamba_forward = mamba_ssm.Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_backward = mamba_ssm.Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            fwd = self.mamba_forward(x)
            x_rev = torch.flip(x, dims=[1])
            rev = self.mamba_backward(x_rev)
            rev = torch.flip(rev, dims=[1])
            return self.norm(fwd + rev)
else:
    class BidirectionalMambaBlock(nn.Module):
        def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.d_inner = int(d_model * expand)
            self.in_proj = nn.Linear(d_model, self.d_inner * 2)
            self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
            self.out_proj = nn.Linear(self.d_inner, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x_proj = self.in_proj(x)
            x_conv, x_ssm = x_proj.chunk(2, dim=-1)
            x_conv = x_conv.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :x.size(1)]
            x_conv = F.silu(x_conv).transpose(1, 2)
            x_out = self.out_proj(x_conv + x_ssm)
            return residual + x_out


class BiMamba(nn.Module):
    """
    BiMamba with True Day-type Gating for Daily Pattern Adjustment.
    支持工作日/周末/小时/分钟等多维时间特征的门控调节。
    """
    def __init__(self, input_dim=1, n_appliances=8, d_model=64, day_type_dim=5):
        super().__init__()
        self.n_appliances = n_appliances
        self.d_model = d_model
        self.day_type_dim = day_type_dim

        # 3通道输入投影 (power, diff, log)
        self.input_proj = nn.Linear(3, d_model)

        # 局部卷积特征提取
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            ResidualConvBlock(d_model)
        )

        # 3层双向Mamba
        self.mamba_layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(3)
        ])

        # ========== 真正的 Day-type Gating ==========
        # 日期特征投影层
        self.day_type_proj = nn.Sequential(
            nn.Linear(day_type_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 门控网络：融合Mamba特征和日期特征
        self.day_type_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid()  # 输出 [0,1]，工作日/周末不同时段门控值不同
        )
        
        # 可学习缩放因子，控制门控强度
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # 可选的残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_appliances)
        )

    def forward(self, x, total_raw=None, sample_positions=None, day_type=None):
        """
        Args:
            x: 归一化总功率 [B, T, 1]
            total_raw: 原始总功率 [B, T]
            sample_positions: 样本起始位置
            day_type: 日期类型特征 [B, T, day_type_dim]
                      包含: [weekday_norm, is_weekend, is_weekday, hour_norm, minute_norm]
        """
        power = x.squeeze(-1)

        # 计算差分和对数特征
        if total_raw is not None:
            diff_raw = torch.diff(total_raw, dim=1, prepend=total_raw[:, :1])
            log_raw = torch.log1p(torch.clamp(total_raw, min=0))
        else:
            diff_raw = torch.zeros_like(power)
            log_raw = torch.zeros_like(power)

        diff_mean = diff_raw.mean(dim=1, keepdim=True)
        diff_std = diff_raw.std(dim=1, keepdim=True) + 1e-8
        diff = (diff_raw - diff_mean) / diff_std

        log_mean = log_raw.mean(dim=1, keepdim=True)
        log_std = log_raw.std(dim=1, keepdim=True) + 1e-8
        log_val = (log_raw - log_mean) / log_std

        # 3通道特征
        h_3ch = torch.stack([power, diff, log_val], dim=-1)
        h = self.input_proj(h_3ch)

        # 局部卷积
        h = h.transpose(1, 2)
        h = self.local_conv(h)
        h = h.transpose(1, 2)

        # 双向Mamba层
        for mamba_layer in self.mamba_layers:
            h = mamba_layer(h)

        # ========== Day-type Gating (Daily Pattern Adjustment) ==========
        if day_type is not None:
            # 将日期特征投影到相同空间
            day_embed = self.day_type_proj(day_type)  # [B, T, d_model]
            
            # 融合特征和日期信息
            combined = torch.cat([h, day_embed], dim=-1)
            gate = self.day_type_gate(combined)  # [B, T, d_model]
            
            # 应用门控：工作日/周末/不同时段门控值自动调整
            # 公式: h_out = h * (1 + scale * (2*gate - 1))
            h = h * (1 + self.gate_scale * (2 * gate - 1))
            
            # 可选：添加残差连接，保留原始信息
            h = h + self.residual_weight * day_embed
        else:
            # 兼容模式：无日期特征时使用自门控
            gate = self.day_type_gate(torch.cat([h, h], dim=-1))
            h = h * (1 + self.gate_scale * (2 * gate - 1))

        # 输出
        output = self.output_proj(h)
        return output, None, None, None


class LSTMExpert(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class TCNExpert(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(DilatedConvBlock(input_dim, hidden_dim, 3, dilation=1), DilatedConvBlock(hidden_dim, hidden_dim, 3, dilation=2), DilatedConvBlock(hidden_dim, hidden_dim, 3, dilation=4))
    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)

class CNNExpert(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, 3, padding=1), nn.ReLU(), nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU())
    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

class TransformerExpert(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    def forward(self, x):
        return self.transformer(self.proj(x))

class DNNExpert(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
    def forward(self, x):
        return self.fc(x)


class FrankWolfeOptimizer:
    def __init__(self, n_tasks=2):
        self.n_tasks = n_tasks
    
    def compute_optimal_weights(self, gradients):
        grad_stack = torch.stack(gradients, dim=0)
        M = torch.mm(grad_stack, grad_stack.t())
        
        alpha = torch.ones(self.n_tasks, device=grad_stack.device) / self.n_tasks
        for _ in range(20):
            grad = 2 * M @ alpha
            idx_min = torch.argmin(grad)
            s = torch.zeros(self.n_tasks, device=grad_stack.device)
            s[idx_min] = 1
            
            d = s - alpha
            gamma = - (d @ M @ alpha) / (d @ M @ d + 1e-8)
            gamma = torch.clamp(gamma, 0, 1)
            
            alpha = (1 - gamma) * alpha + gamma * s
        
        return alpha


class HMoE(nn.Module):
    def __init__(self, input_dim=1, n_appliances=8, hidden_dim=64):
        super().__init__()
        self.n_appliances = n_appliances
        self.hidden_dim = hidden_dim
        
        self.experts = nn.ModuleList([
            LSTMExpert(input_dim, hidden_dim),
            TCNExpert(input_dim, hidden_dim),
            CNNExpert(input_dim, hidden_dim),
            TransformerExpert(input_dim, hidden_dim),
            DNNExpert(input_dim, hidden_dim)
        ])
        
        self.gate_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.experts)),
            nn.Softmax(dim=-1)
        )
        self.gate_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.experts)),
            nn.Softmax(dim=-1)
        )
        
        tower_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        
        self.tower_a = nn.Sequential(
            nn.TransformerEncoder(tower_encoder_layer, num_layers=3),
            nn.Linear(hidden_dim, n_appliances),
            nn.Sigmoid()
        )
        
        self.tower_b = nn.Sequential(
            nn.TransformerEncoder(tower_encoder_layer, num_layers=3),
            nn.Linear(hidden_dim, n_appliances),
            nn.ReLU()
        )
        
        self.power_scale = nn.Parameter(torch.ones(n_appliances) * 500)
        
        self.use_frank_wolfe = False
    
    def forward(self, x, total_raw=None, sample_positions=None):
        batch_size, seq_len = x.size(0), x.size(1)
        gate_input = x.mean(dim=1)
        gate_weights_a = self.gate_a(gate_input)
        gate_weights_b = self.gate_b(gate_input)
        
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)
            if out.size(1) != seq_len:
                if out.size(1) > seq_len:
                    out = out[:, :seq_len, :]
                else:
                    out = F.pad(out, (0, 0, 0, seq_len - out.size(1)))
            expert_outputs.append(out)
        
        expert_stack = torch.stack(expert_outputs, dim=2)
        
        gate_weights_a = gate_weights_a.view(batch_size, 1, -1, 1)
        gate_weights_b = gate_weights_b.view(batch_size, 1, -1, 1)
        
        feat_a = (expert_stack * gate_weights_a).sum(dim=2)
        feat_b = (expert_stack * gate_weights_b).sum(dim=2)
        
        state_out = self.tower_a(feat_a)
        power_out = self.tower_b(feat_b) * self.power_scale
        
        return power_out, state_out, power_out, None


def contrastive_loss(features, temperature=0.1):
    if features is None:
        return torch.tensor(0.0)
    
    if features.dim() == 3:
        features = features.mean(dim=1)
    
    features = F.normalize(features, dim=1)
    similarity = torch.mm(features, features.t()) / temperature
    batch_size = features.size(0)
    labels = torch.arange(batch_size, device=features.device)
    loss = F.cross_entropy(similarity, labels)
    return loss


def mamba2nilm_loss(output, target_raw, tau=0.1, lam=0.001):
    """Mamba2NILM损失函数 - 论文Eq.(9)
    L = MSE + KL(softmax(y_hat/tau) || softmax(y/tau)) + soft-margin + lambda * L1
    含有NaN masking: 跳过target中NaN的位置
    """
    valid = ~torch.isnan(target_raw)
    if not valid.any():
        return (output * 0).sum()

    mse = F.mse_loss(output[valid], target_raw[valid])

    y_min = min(target_raw[valid].min().item(), output[valid].min().item())
    y_max = max(target_raw[valid].max().item(), output[valid].max().item())
    if y_max - y_min > 1e-6:
        y_hat_norm = (output - y_min) / (y_max - y_min + 1e-8)
        y_norm = (target_raw - y_min) / (y_max - y_min + 1e-8)
        y_norm = torch.where(torch.isnan(y_norm), output.detach(), y_norm)
        log_p = F.log_softmax(y_hat_norm.view(-1, y_hat_norm.size(-1)) / tau, dim=-1)
        p_target = F.softmax(y_norm.view(-1, y_norm.size(-1)) / tau, dim=-1)
        kl = F.kl_div(log_p, p_target, reduction='batchmean')
    else:
        kl = torch.tensor(0.0, device=output.device)

    state_target = (target_raw > 10).float()
    state_target_bi = 2 * state_target - 1
    state_pred = 2 * torch.sigmoid(output) - 1
    soft_margin = torch.mean(torch.log(1 + torch.exp(-state_target_bi * state_pred)))

    l1 = F.l1_loss(output[valid], target_raw[valid])

    return mse + 0.1 * kl + 0.1 * soft_margin + lam * l1


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def train_model(model, train_loader, val_loader, n_epochs, lr, device, 
                model_name='Model', save_path=None, loss_type='mse', log_prefix=""):
    ckpt_path = save_path.replace('.pt', '_ckpt.pt') if save_path else None
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    disable_amp_models = ['DU-NILM', 'DUNILM', 'Mamba2NILM', 'BiMamba']
    use_amp = device.type == 'cuda' and not any(m in model_name for m in disable_amp_models)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_score = float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')
    best_epoch = 0
    epoch_times = []
    max_memory = 0
    start_epoch = 0
    recent_val_maes = []
    
    # 检查模型是否支持 day_type 参数（BiMamba 支持）
    supports_day_type = 'day_type' in model.forward.__code__.co_varnames
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"{log_prefix}  Resuming from checkpoint...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if use_amp and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch']
        best_score = ckpt.get('best_score', float('inf'))
        best_mae = ckpt.get('best_mae', float('inf'))
        best_rmse = ckpt.get('best_rmse', float('inf'))
        best_epoch = ckpt.get('best_epoch', 0)
        epoch_times = ckpt.get('epoch_times', [])
        max_memory = ckpt.get('max_memory', 0)
        recent_val_maes = ckpt.get('recent_val_maes', [])
        print(f"{log_prefix}  Resumed from epoch {start_epoch}, best MAE={best_mae:.2f}W", flush=True)
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_start = time.time()
        
        for batch in train_loader:
            # 动态解包，兼容 4 或 5 个返回值
            if len(batch) == 5:
                total_input, total_raw, target_raw, sample_positions, day_type = batch
            else:
                total_input, total_raw, target_raw, sample_positions = batch
                day_type = None
            
            total_input = total_input.to(device, non_blocking=use_amp)
            total_raw = total_raw.to(device, non_blocking=use_amp)
            target_raw = target_raw.to(device, non_blocking=use_amp)
            sample_positions = sample_positions.to(device, non_blocking=use_amp).squeeze(-1)
            if day_type is not None:
                day_type = day_type.to(device, non_blocking=use_amp)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # 根据模型是否支持 day_type 选择调用方式
                    if supports_day_type and day_type is not None:
                        output, feat2, feat3, contr_loss = model(total_input, total_raw, sample_positions, day_type=day_type)
                    else:
                        output, feat2, feat3, contr_loss = model(total_input, total_raw, sample_positions)
                    
                    if hasattr(model, 'use_frank_wolfe') and model.use_frank_wolfe:
                        state_out = feat2
                        power_out = feat3
                        
                        state_target = (target_raw > 10).float()
                        state_target_bi = 2 * state_target - 1
                        
                        L_c = torch.mean(torch.log(1 + torch.exp(-state_target_bi * (2 * state_out - 1))))
                        
                        mse_loss = F.mse_loss(power_out, target_raw)
                        l1_loss = F.l1_loss(power_out, target_raw)
                        
                        L_r = mse_loss + 0.1 * l1_loss
                        
                        if not hasattr(model, 'fw_optimizer'):
                            model.fw_optimizer = FrankWolfeOptimizer(n_tasks=2)
                        
                        grad_L_c = torch.autograd.grad(L_c, model.parameters(), retain_graph=True, create_graph=False, allow_unused=True)
                        grad_L_r = torch.autograd.grad(L_r, model.parameters(), retain_graph=True, create_graph=False, allow_unused=True)
                        
                        grad_L_c_flat = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) 
                                                   for g, p in zip(grad_L_c, model.parameters())])
                        grad_L_r_flat = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) 
                                                   for g, p in zip(grad_L_r, model.parameters())])
                        
                        gradients = [grad_L_c_flat, grad_L_r_flat]
                        alphas = model.fw_optimizer.compute_optimal_weights(gradients)
                        
                        alpha_c = alphas[0]
                        alpha_r = alphas[1]
                        
                        loss = alpha_c * L_c + alpha_r * L_r
                    else:
                        if hasattr(model, 'use_mamba2nilm_loss'):
                            loss = mamba2nilm_loss(output, target_raw)
                        elif output.dim() == 2:
                            center_idx = target_raw.shape[1] // 2
                            target_center = target_raw[:, center_idx, :]
                            valid = ~torch.isnan(target_center)
                            if valid.any():
                                if hasattr(model, 'loss_type') and model.loss_type == 'mae':
                                    loss = F.l1_loss(output[valid], target_center[valid])
                                else:
                                    loss = F.mse_loss(output[valid], target_center[valid])
                            else:
                                loss = F.mse_loss(output, output)
                        else:
                            if hasattr(model, 'loss_type') and model.loss_type == 'mae':
                                valid = ~torch.isnan(target_raw)
                                if valid.any():
                                    loss = F.l1_loss(output[valid], target_raw[valid])
                                else:
                                    loss = F.l1_loss(output, output)
                            else:
                                valid = ~torch.isnan(target_raw)
                                if valid.any():
                                    mse_loss = F.mse_loss(output[valid], target_raw[valid])
                                else:
                                    mse_loss = F.mse_loss(output, output)
                                if contr_loss is not None:
                                    loss = mse_loss + 1.0 * contr_loss
                                elif feat2 is not None:
                                    contr_loss = contrastive_loss(feat2)
                                    loss = mse_loss + 1.0 * contr_loss
                                else:
                                    loss = mse_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                use_fw = hasattr(model, 'use_frank_wolfe') and model.use_frank_wolfe
                if not use_fw:
                    max_norm = 10.0 if hasattr(model, 'use_mamba2nilm_loss') else 1.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 根据模型是否支持 day_type 选择调用方式
                if supports_day_type and day_type is not None:
                    output, feat2, feat3, contr_loss = model(total_input, total_raw, sample_positions, day_type=day_type)
                else:
                    output, feat2, feat3, contr_loss = model(total_input, total_raw, sample_positions)
                
                if hasattr(model, 'use_frank_wolfe') and model.use_frank_wolfe:
                    state_out = feat2
                    power_out = feat3
                    
                    state_target = (target_raw > 10).float()
                    
                    state_target_bi = 2 * state_target - 1
                    
                    L_c = torch.mean(torch.log(1 + torch.exp(-state_target_bi * (2 * state_out - 1))))
                    
                    mse_loss = F.mse_loss(power_out, target_raw)
                    l1_loss = F.l1_loss(power_out, target_raw)
                    
                    L_r = mse_loss + 0.1 * l1_loss
                    
                    if not hasattr(model, 'fw_optimizer'):
                        model.fw_optimizer = FrankWolfeOptimizer(n_tasks=2)
                    
                    grad_L_c = torch.autograd.grad(L_c, model.parameters(), retain_graph=True, create_graph=False, allow_unused=True)
                    grad_L_r = torch.autograd.grad(L_r, model.parameters(), retain_graph=True, create_graph=False, allow_unused=True)
                    
                    grad_L_c_flat = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) 
                                               for g, p in zip(grad_L_c, model.parameters())])
                    grad_L_r_flat = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) 
                                               for g, p in zip(grad_L_r, model.parameters())])
                    
                    gradients = [grad_L_c_flat, grad_L_r_flat]
                    alphas = model.fw_optimizer.compute_optimal_weights(gradients)
                    
                    alpha_c = alphas[0]
                    alpha_r = alphas[1]
                    
                    loss = alpha_c * L_c + alpha_r * L_r
                else:
                    if hasattr(model, 'use_mamba2nilm_loss'):
                        loss = mamba2nilm_loss(output, target_raw)
                    elif output.dim() == 2:
                        center_idx = target_raw.shape[1] // 2
                        target_center = target_raw[:, center_idx, :]
                        valid = ~torch.isnan(target_center)
                        if valid.any():
                            if hasattr(model, 'loss_type') and model.loss_type == 'mae':
                                loss = F.l1_loss(output[valid], target_center[valid])
                            else:
                                loss = F.mse_loss(output[valid], target_center[valid])
                        else:
                            loss = F.mse_loss(output, output)
                    else:
                        if hasattr(model, 'loss_type') and model.loss_type == 'mae':
                            valid = ~torch.isnan(target_raw)
                            if valid.any():
                                loss = F.l1_loss(output[valid], target_raw[valid])
                            else:
                                loss = F.l1_loss(output, output)
                        else:
                            valid = ~torch.isnan(target_raw)
                            if valid.any():
                                mse_loss = F.mse_loss(output[valid], target_raw[valid])
                            else:
                                mse_loss = F.mse_loss(output, output)
                            if contr_loss is not None:
                                loss = mse_loss + 1.0 * contr_loss
                            elif feat2 is not None:
                                contr_loss = contrastive_loss(feat2)
                                loss = mse_loss + 1.0 * contr_loss
                            else:
                                loss = mse_loss
                loss.backward()
                use_fw = hasattr(model, 'use_frank_wolfe') and model.use_frank_wolfe
                if not use_fw:
                    max_norm = 10.0 if hasattr(model, 'use_mamba2nilm_loss') else 1.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            
            mem = get_memory_usage()
            if mem > max_memory:
                max_memory = mem
        
        scheduler.step()
        
        model.eval()
        total_abs_error = 0
        total_squared_error = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 动态解包验证集
                if len(batch) == 5:
                    total_input, total_raw, target_raw, sample_positions, day_type = batch
                else:
                    total_input, total_raw, target_raw, sample_positions = batch
                    day_type = None
                
                total_input = total_input.to(device, non_blocking=use_amp)
                total_raw = total_raw.to(device, non_blocking=use_amp)
                if day_type is not None:
                    day_type = day_type.to(device, non_blocking=use_amp)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        if supports_day_type and day_type is not None:
                            output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device), day_type=day_type)
                        else:
                            output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device))
                else:
                    if supports_day_type and day_type is not None:
                        output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device), day_type=day_type)
                    else:
                        output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device))
                
                if hasattr(model, 'use_frank_wolfe') and model.use_frank_wolfe and feat3 is not None:
                    output = feat3.float().cpu().numpy()
                else:
                    output = output.float().cpu().numpy()
                target = target_raw.numpy()
                
                if output.ndim == 2:
                    center_idx = target.shape[1] // 2
                    target = target[:, center_idx, :]
                
                valid_mask = ~np.isnan(target)
                if valid_mask.any():
                    total_samples += valid_mask.sum()
                    total_abs_error += np.sum(np.abs(output[valid_mask] - target[valid_mask]))
                    total_squared_error += np.sum((output[valid_mask] - target[valid_mask]) ** 2)
        
        mae = total_abs_error / total_samples
        rmse = np.sqrt(total_squared_error / total_samples)
        score = 0.6 * mae + 0.4 * rmse
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if score < best_score:
            best_score = score
            best_mae = mae
            best_rmse = rmse
            best_epoch = epoch + 1
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        print(f"{log_prefix}  Epoch {epoch + 1}/{n_epochs} | Val MAE: {mae:.2f}W | Val RMSE: {rmse:.2f}W | Time: {epoch_time:.1f}s", flush=True)
        
        recent_val_maes.append(mae)
        if len(recent_val_maes) > 4:
            recent_val_maes.pop(0)
        
        if ckpt_path:
            ckpt_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'best_mae': best_mae,
                'best_rmse': best_rmse,
                'best_epoch': best_epoch,
                'epoch_times': epoch_times,
                'n_epochs': n_epochs,
                'max_memory': max_memory,
                'recent_val_maes': recent_val_maes,
            }
            if use_amp and scaler is not None:
                ckpt_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)
        
        if len(recent_val_maes) == 4:
            max_recent_mae = max(recent_val_maes[:-1])
            if mae > max_recent_mae - 0.1:
                print(f"{log_prefix}  Early stopping triggered! Val MAE {mae:.2f}W > {max_recent_mae:.2f}W - 0.1", flush=True)
                break
    
    avg_time = np.mean(epoch_times)
    print(f"{log_prefix}  Best: MAE={best_mae:.2f}W, RMSE={best_rmse:.2f}W at epoch {best_epoch}", flush=True)
    
    return {
        'mae': float(best_mae), 
        'rmse': float(best_rmse), 
        'score': float(best_score), 
        'epoch': best_epoch, 
        'avg_epoch_time': float(avg_time), 
        'max_memory_mb': float(max_memory),
        'epoch_times': [float(t) for t in epoch_times]
    }


def evaluate_model(model, test_loader, device, n_appliances=N_APPLIANCES, model_name=""):
    model.eval()
    disable_amp_models = ['DU-NILM', 'DUNILM', 'Mamba2NILM', 'BiMamba']
    use_amp = device.type == 'cuda' and not any(m in model_name for m in disable_amp_models)
    
    # 检查模型是否支持 day_type 参数（BiMamba 支持）
    supports_day_type = 'day_type' in model.forward.__code__.co_varnames
    
    app_abs_error = np.zeros(n_appliances)
    app_squared_error = np.zeros(n_appliances)
    total_samples = 0
    test_start = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            # 动态解包，兼容 4 或 5 个返回值
            if len(batch) == 5:
                total_input, total_raw, target_raw, sample_positions, day_type = batch
            else:
                total_input, total_raw, target_raw, sample_positions = batch
                day_type = None
            
            total_input = total_input.to(device, non_blocking=use_amp)
            total_raw = total_raw.to(device, non_blocking=use_amp)
            if day_type is not None:
                day_type = day_type.to(device, non_blocking=use_amp)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    if supports_day_type and day_type is not None:
                        output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device), day_type=day_type)
                    else:
                        output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device))
            else:
                if supports_day_type and day_type is not None:
                    output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device), day_type=day_type)
                else:
                    output, feat2, feat3, _ = model(total_input, total_raw, sample_positions.squeeze(-1).to(device))
            
            if hasattr(model, 'use_frank_wolfe') and model.use_frank_wolfe and feat3 is not None:
                output = feat3.float().cpu().numpy()
            else:
                output = output.float().cpu().numpy()
            target = target_raw.numpy()
            
            if output.ndim == 2:
                center_idx = target.shape[1] // 2
                target = target[:, center_idx, :]
                batch_samples = output.shape[0]
                total_samples += batch_samples
                
                for i in range(n_appliances):
                    valid = ~np.isnan(target[:, i])
                    if valid.any():
                        app_abs_error[i] += np.sum(np.abs(output[valid, i] - target[valid, i]))
                        app_squared_error[i] += np.sum((output[valid, i] - target[valid, i]) ** 2)
            else:
                batch_samples = output.shape[0] * output.shape[1]
                total_samples += batch_samples
                
                for i in range(n_appliances):
                    valid = ~np.isnan(target[:, :, i])
                    if valid.any():
                        app_abs_error[i] += np.sum(np.abs(output[:, :, i][valid] - target[:, :, i][valid]))
                        app_squared_error[i] += np.sum((output[:, :, i][valid] - target[:, :, i][valid]) ** 2)
    
    test_time = time.time() - test_start
    
    app_mae = app_abs_error / total_samples
    app_rmse = np.sqrt(app_squared_error / total_samples)
    total_mae = np.mean(app_mae)
    total_rmse = np.sqrt(np.mean(app_squared_error / total_samples))
    
    return {
        'total_mae': float(total_mae),
        'total_rmse': float(total_rmse),
        'app_mae': app_mae.tolist(),
        'app_rmse': app_rmse.tolist(),
        'test_time': float(test_time),
        'test_samples': int(total_samples)
    }


def train_per_appliance_method(name, ModelClass, device, n_epochs=20, lr=1e-4, log_prefix=""):
    print(f"{log_prefix}[{name}] - Per-appliance training", flush=True)
    
    all_app_mae = np.zeros(N_APPLIANCES)
    all_app_rmse = np.zeros(N_APPLIANCES)
    all_epoch_times = []
    all_max_memory = 0
    total_params = 0
    test_time_total = 0
    
    for app_idx in range(N_APPLIANCES):
        app_name = APPLIANCE_NAMES[app_idx]
        save_path = os.path.join(RESULT_DIR, f'best_{name}_app{app_idx}_v2.pt')
        
        # 启用日期类型特征
        train_ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='train', 
                               appliance_idx=app_idx, use_day_type=True)
        val_ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='val', 
                             appliance_idx=app_idx, use_day_type=True)
        test_ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='test', 
                              appliance_idx=app_idx, use_day_type=True)
        
        train_loader_app = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_app = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_app = DataLoader(test_ds, batch_size=70, shuffle=False, num_workers=4, pin_memory=False)
        
        model = ModelClass(n_appliances=1, day_type_dim=5).to(device)  # 传入 day_type_dim
        
        if app_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
        
        print(f"{log_prefix}  [{app_name}] Training...", flush=True)
        result = train_model(model, train_loader_app, val_loader_app, n_epochs, lr, device, 
                            f'{name}_{app_name}', save_path, loss_type='mse', log_prefix=log_prefix)
        
        all_epoch_times.extend(result.get('epoch_times', []))
        if result.get('max_memory_mb', 0) > all_max_memory:
            all_max_memory = result['max_memory_mb']
        
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        eval_result = evaluate_model(model, test_loader_app, device, n_appliances=1, model_name=name)
        
        all_app_mae[app_idx] = eval_result['app_mae'][0]
        all_app_rmse[app_idx] = eval_result['app_rmse'][0]
        test_time_total += eval_result['test_time']
        
        print(f"{log_prefix}  [{app_name}] MAE={all_app_mae[app_idx]:.2f}W, RMSE={all_app_rmse[app_idx]:.2f}W", flush=True)
        
        del model, train_ds, val_ds, test_ds
        gc.collect()
        torch.cuda.empty_cache()
    
    total_mae = np.mean(all_app_mae)
    total_rmse = np.sqrt(np.mean(all_app_rmse ** 2))
    avg_time = np.mean(all_epoch_times) if all_epoch_times else 0
    
    print(f"{log_prefix}[{name}] Overall: MAE={total_mae:.2f}W, RMSE={total_rmse:.2f}W", flush=True)
    
    return {
        'total_mae': float(total_mae),
        'total_rmse': float(total_rmse),
        'app_mae': all_app_mae.tolist(),
        'app_rmse': all_app_rmse.tolist(),
        'avg_epoch_time': float(avg_time),
        'max_memory_mb': float(all_max_memory),
        'test_time': float(test_time_total),
        'params': int(total_params * N_APPLIANCES),
        'is_per_appliance': True
    }


def train_multi_output_method(name, ModelClass, train_loader, val_loader, test_loader, device, n_epochs=40, lr=1e-4, log_prefix=""):
    print(f"{log_prefix}[{name}] - Multi-output training", flush=True)
    
    save_path = os.path.join(RESULT_DIR, f'best_{name}_v2.pt')
    ckpt_path = os.path.join(RESULT_DIR, f'best_{name}_v2_ckpt.pt')
    
    model = ModelClass(n_appliances=N_APPLIANCES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    result = train_model(model, train_loader, val_loader, n_epochs, lr, device, 
                        name, save_path, loss_type='mse', log_prefix=log_prefix)
    
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    eval_result = evaluate_model(model, test_loader, device, model_name=name)
    
    return {
        'total_mae': eval_result['total_mae'],
        'total_rmse': eval_result['total_rmse'],
        'app_mae': eval_result['app_mae'],
        'app_rmse': eval_result['app_rmse'],
        'avg_epoch_time': result['avg_epoch_time'],
        'max_memory_mb': result['max_memory_mb'],
        'test_time': eval_result['test_time'],
        'params': int(total_params),
        'is_per_appliance': False
    }


def main(methods=None, prefix="[BASELINE] ", force_train=False):
    print(f"{prefix}" + "=" * 70, flush=True)
    print(f"{prefix}NILM Baseline Experiments - Unified Data Split", flush=True)
    print(f"{prefix}Train: 0-{TRAIN_SIZE}, Val: {TRAIN_SIZE}-{TRAIN_SIZE+VAL_SIZE}, Test: rest", flush=True)
    print(f"{prefix}" + "=" * 70, flush=True)
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{prefix}Device: {device}", flush=True)
    
    train_ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='train')
    val_ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='val')
    test_ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    per_appliance_methods = {
        'Seq2Point': Seq2PointSeq2Seq,
        'DU-NILM': DUNILM,
        'FFSTT': FFSTT,
        'Attention-NILM': AttentionNILM,
        'Seq2Seq': Seq2Seq,
    }
    
    multi_output_methods = {
        'CL-MDTFN': CLMDTFN,
        'HMoE': HMoE,
    }

    multi_output_methods['BiMamba'] = BiMamba

    if _HAS_MAMBA:
        multi_output_methods['Mamba2NILM'] = Mamba2NILM
    
    if methods is None:
        methods = list(per_appliance_methods.keys()) + list(multi_output_methods.keys())
    
    all_results = {}
    
    for method in methods:
        if method in per_appliance_methods:
            ModelClass = per_appliance_methods[method]
            result = train_per_appliance_method(method, ModelClass, device, n_epochs=40, log_prefix=prefix)
            all_results[method] = result
        elif method in multi_output_methods:
            ModelClass = multi_output_methods[method]
            n_epochs = 40 if method == 'Mamba2NILM' else 40
            lr = 1e-4 if method == 'Mamba2NILM' else 1e-4
            result = train_multi_output_method(method, ModelClass, train_loader, val_loader, test_loader,
                                               device, n_epochs=n_epochs, lr=lr, log_prefix=prefix)
            all_results[method] = result
    
    print(f"\n{prefix}" + "=" * 70, flush=True)
    print(f"{prefix}RESULTS SUMMARY", flush=True)
    print(f"{prefix}" + "=" * 70, flush=True)
    
    print(f"\n{prefix}{'Method':<18} {'MAE(W)':>10} {'RMSE(W)':>10} {'Params':>10} {'AvgTime(s)':>12} {'TestTime(s)':>12}", flush=True)
    print(f"{prefix}" + "-" * 72, flush=True)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total_mae'])
    for name, r in sorted_results:
        print(f"{prefix}{name:<18} {r['total_mae']:>10.2f} {r['total_rmse']:>10.2f} {r['params']:>10} {r['avg_epoch_time']:>12.1f} {r['test_time']:>12.1f}", flush=True)
    
    print(f"\n{prefix}Per-Appliance Results:", flush=True)
    print(f"{prefix}{'Method':<18} " + " ".join([f"{name:>8}" for name in APPLIANCE_NAMES]), flush=True)
    print(f"{prefix}" + "-" * 86, flush=True)
    
    for name, r in sorted_results:
        mae_str = " ".join([f"{mae:>8.2f}" for mae in r['app_mae']])
        print(f"{prefix}{name:<18} {mae_str}", flush=True)
    
    results_file = os.path.join(RESULT_DIR, 'baseline_results_v2.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{prefix}Results saved to {results_file}", flush=True)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NILM Baseline Experiments V2')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to train')
    parser.add_argument('--prefix', type=str, default='[BASELINE] ',
                        help='Log prefix')
    parser.add_argument('--force-train', action='store_true',
                        help='Force training')
    args = parser.parse_args()
    
    try:
        results = main(methods=args.methods, prefix=args.prefix, force_train=args.force_train)
        print(f"\n{args.prefix}Baseline experiments completed!", flush=True)
    except Exception as e:
        print(f"\n{args.prefix}Experiment failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
