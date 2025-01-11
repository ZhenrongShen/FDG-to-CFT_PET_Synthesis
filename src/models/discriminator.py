import torch
import torch.nn as nn
from .modules import DownBlock, SN_Conv3d, SN_Linear


class Discriminator(nn.Module):
    def __init__(self,
                 ndf: int = 16,
                 num_class: int = 2,
                 use_spectral_norm: bool = True):
        super().__init__()

        # Feature channels at each spatial resolution
        nfc_multi = {8: 8, 16: 4, 32: 2, 64: 1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        # Network structure
        self.to_map = nn.Sequential(
            SN_Conv3d(1, nfc[64], 3, 1, 1, bias=False),
            DownBlock(nfc[64], nfc[32], use_spectral_norm=use_spectral_norm),
            DownBlock(nfc[32], nfc[16], use_spectral_norm=use_spectral_norm),
            DownBlock(nfc[16], nfc[8], use_spectral_norm=use_spectral_norm),
        )

        # Projection head
        if use_spectral_norm:
            self.to_logits = SN_Conv3d(nfc[8], 1, 1, 1, 0, bias=False)
            self.to_cls_embed = SN_Linear(num_class, nfc[8], bias=False)
        else:
            self.to_logits = nn.Conv3d(nfc[8], 1, 1, 1, 0, bias=False)
            self.to_cls_embed = nn.Linear(num_class, nfc[8], bias=False)

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        """
        :param x: is the input with shape `[batch_size, in_channels, depth, height, width]`
        :param label: is the class label with shape `[batch_size, num_class]`
        """
        # Main branch
        feat_out = self.to_map(x)
        logits = self.to_logits(feat_out)

        # Projection head
        cls_embed = self.to_cls_embed(label).view(label.shape[0], -1, 1, 1, 1)
        logits += torch.sum(cls_embed * feat_out, dim=1, keepdim=True)

        return logits
