"""SalsaNext (Cortinhal et al., 2020, arXiv:2003.03653): range-image lidar segmentation.

Architecture as in the paper's reference implementation, without the Bayesian uncertainty
head. Four 2x poolings, so the range image's height and width must be divisible by 16
(the SICK's 16 layers give a single row at the bottleneck).
"""
import torch
import torch.nn.functional as F
from torch import nn


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, 1)
        self.conv2 = nn.Conv2d(out_filters, out_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, 3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        shortcut = self.act(self.conv1(x))
        res = self.bn1(self.act(self.conv2(shortcut)))
        res = self.bn2(self.act(self.conv3(res)))
        return shortcut + res


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout, pooling=True, drop_out=True):
        super().__init__()
        self.pooling, self.drop_out = pooling, drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, 1)
        self.conv2 = nn.Conv2d(in_filters, out_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, 3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv4 = nn.Conv2d(out_filters, out_filters, 2, dilation=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, 1)
        self.bn4 = nn.BatchNorm2d(out_filters)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        shortcut = self.act(self.conv1(x))
        a1 = self.bn1(self.act(self.conv2(x)))
        a2 = self.bn2(self.act(self.conv3(a1)))
        a3 = self.bn3(self.act(self.conv4(a2)))
        res = shortcut + self.bn4(self.act(self.conv5(torch.cat((a1, a2, a3), dim=1))))
        out = self.dropout(res) if self.drop_out else res
        if self.pooling:
            return self.pool(out), res
        return out


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout, drop_out=True):
        super().__init__()
        self.drop_out = drop_out
        self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout2d(dropout) for _ in range(3))
        self.conv1 = nn.Conv2d(in_filters // 4 + 2 * out_filters, out_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, 3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, 2, dilation=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, 1)
        self.bn4 = nn.BatchNorm2d(out_filters)
        self.act = nn.LeakyReLU()

    def forward(self, x, skip):
        up = F.pixel_shuffle(x, 2)
        if self.drop_out:
            up = self.dropout1(up)
        up = torch.cat((up, skip), dim=1)
        if self.drop_out:
            up = self.dropout2(up)
        e1 = self.bn1(self.act(self.conv1(up)))
        e2 = self.bn2(self.act(self.conv2(e1)))
        e3 = self.bn3(self.act(self.conv3(e2)))
        out = self.bn4(self.act(self.conv4(torch.cat((e1, e2, e3), dim=1))))
        return self.dropout3(out) if self.drop_out else out


class SalsaNext(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 5):
        super().__init__()
        self.context = nn.Sequential(ResContextBlock(in_channels, 32), ResContextBlock(32, 32),
                                     ResContextBlock(32, 32))
        self.res1 = ResBlock(32, 64, 0.2, pooling=True, drop_out=False)
        self.res2 = ResBlock(64, 128, 0.2, pooling=True)
        self.res3 = ResBlock(128, 256, 0.2, pooling=True)
        self.res4 = ResBlock(256, 256, 0.2, pooling=True)
        self.res5 = ResBlock(256, 256, 0.2, pooling=False)
        self.up1 = UpBlock(256, 128, 0.2)
        self.up2 = UpBlock(128, 128, 0.2)
        self.up3 = UpBlock(128, 64, 0.2)
        self.up4 = UpBlock(64, 32, 0.2, drop_out=False)
        self.logits = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.context(x)
        d0, s0 = self.res1(x)
        d1, s1 = self.res2(d0)
        d2, s2 = self.res3(d1)
        d3, s3 = self.res4(d2)
        x = self.res5(d3)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return self.logits(x)  # raw logits; softmax in the loss / at inference


def lovasz_softmax(probs, labels, ignore=255):
    """Lovasz-Softmax loss (Berman et al., 2018) over the classes present in `labels`."""
    c = probs.shape[1]
    probs = probs.permute(0, 2, 3, 1).reshape(-1, c)
    labels = labels.reshape(-1)
    keep = labels != ignore
    probs, labels = probs[keep], labels[keep]
    if labels.numel() == 0:
        return probs.sum() * 0.0
    losses = []
    for k in labels.unique():
        fg = (labels == k).float()
        errors = (fg - probs[:, k]).abs()
        errors, perm = torch.sort(errors, descending=True)
        fg = fg[perm]
        gts = fg.sum()
        intersection = gts - fg.cumsum(0)
        union = gts + (1 - fg).cumsum(0)
        grad = 1.0 - intersection / union
        grad[1:] = grad[1:] - grad[:-1]
        losses.append(torch.dot(errors, grad))
    return torch.stack(losses).mean()
