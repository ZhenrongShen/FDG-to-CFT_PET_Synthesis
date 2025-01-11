import torch
import torch.nn as nn
from typing import List

from .modules import ResBlock, UpBlock, DownBlock, SN_Conv3d



class Generator(nn.Module):
    def __init__(self, ngf: int, n_res_blocks: int, channel_multipliers: List[int], use_spectral_norm: bool=False):
        super().__init__()

        # Number of channels at each resolution level
        channels_list = [ngf * m for m in channel_multipliers]
        levels = len(channel_multipliers)  # number of levels

        # First 3×3 convolution
        if use_spectral_norm:
            self.in_proj = SN_Conv3d(1, channels_list[0], 3, padding=1)
        else:
            self.in_proj = nn.Conv3d(1, channels_list[0], 3, padding=1)

        # Encoder of the U-Net
        encoder_block_channels = []
        channels = channels_list[0]

        self.encoder_blocks = nn.ModuleList()
        for i in range(levels):
            for j in range(n_res_blocks):
                enc_layers = []
                # Before residual block, downsample at all scales except for the first one
                if i != 0 and j == 0:
                    enc_layers.append(DownBlock(channels, channels, use_spectral_norm=use_spectral_norm))
                # Add residual block: [previous channels --> current channels]
                enc_layers.append(ResBlock(channels, channels_list[i], use_spectral_norm=use_spectral_norm))
                channels = channels_list[i]
                # Add them to the encoder of the U-Net
                self.encoder_blocks.append(nn.Sequential(*enc_layers))
                # Keep track of the channel number of the output
                encoder_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = nn.Sequential(
            ResBlock(channels, channels, use_spectral_norm=True),
            ResBlock(channels, channels, use_spectral_norm=True),
        )

        # Decoder of the U-Net
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(levels)):  # levels in reverse order
            for j in range(n_res_blocks):
                # Add residual block: [previous channels + skip connections --> current channels]
                dec_layers = [ResBlock(channels + encoder_block_channels.pop(), channels_list[i], use_spectral_norm=use_spectral_norm)]
                channels = channels_list[i]
                # After last residual block, up-sample at all levels except for the first one
                if i != 0 and j == n_res_blocks - 1:
                    dec_layers.append(UpBlock(channels, channels, use_spectral_norm=use_spectral_norm))
                # Add them to the decoder of the U-Net
                self.decoder_blocks.append(nn.Sequential(*dec_layers))

        # Final 3×3 convolution
        if use_spectral_norm:
            self.out_proj = nn.Sequential(
                SN_Conv3d(channels, 1, 3, padding=1),
                nn.Tanh(),
            )
        else:
            self.out_proj = nn.Sequential(
                nn.Conv3d(channels, 1, 3, padding=1),
                nn.Tanh(),
            )

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, in_channels, depth, height, width]`
        """
        # First 3×3 convolution
        x = self.in_proj(x)

        # Encoder of the U-Net
        x_input_block = []  # To store the encoder outputs for skip connections
        for module in self.encoder_blocks:
            x = module(x)
            x_input_block.append(x)

        # Middle of the U-Net
        x = self.middle_block(x)

        # Decoder of the U-Net
        for module in self.decoder_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x)

        # Final normalization and 3×3 convolution
        return self.out_proj(x)
