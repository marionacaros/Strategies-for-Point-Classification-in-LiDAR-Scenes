import torch.nn as nn
import torch.nn.functional as F
from src.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

EMBED_DIM = 512


class PointNetAbstractionEncoder(nn.Module):
    def __init__(self):
        super(PointNetAbstractionEncoder, self).__init__()
        # in_channel = 7 input features + 1 normalized z offset added by sample_and_group
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=5 + 3, mlp=[32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 1, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 1, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 1, [256, 256, EMBED_DIM])

    def forward(self, pc):
        l0_points = pc
        l0_xyz = pc[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [b, 3, 1024] [b, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [b, 3, 256] [b, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [b, 3, 64] [b, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [b, 3, 16] [b, 512, 16]

        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points


class PointNet2(nn.Module):
    """PointNet++ SSG segmentation. Input [B, 7, N] (N >= 1024), output log-softmax [B, N, num_classes]."""

    def __init__(self, num_classes):
        super(PointNet2, self).__init__()

        self.encoder = PointNetAbstractionEncoder()

        self.fp4 = PointNetFeaturePropagation(256 + EMBED_DIM, [256, 256])  # 768 channels
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, pc):
        l0_xyz = pc[:, :3, :]
        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points = self.encoder(pc)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [b, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [b, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [b, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [b, 128, N]

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # [b, 128, N]
        x = self.conv2(x)  # [b, n_classes, N]
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # [b, N, n_classes]
        return x, None
