import torch
import torch.nn as nn
import torch.nn.functional as F


class LongTermEncoder(nn.Module):
    def __init__(self, input_channels, out_z=512, ndf=64):
        super(LongTermEncoder, self).__init__()
        self.ndf = ndf
        self.out_z = out_z

        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, ndf, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(ndf, ndf * 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 4, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, ndf * 8, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 8, out_z, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_z),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.residual_connection = nn.Sequential(
            nn.Conv1d(ndf, out_z, kernel_size=1, stride=16, bias=False),
            nn.BatchNorm1d(out_z)
        )

    def forward(self, x):
        x_initial = self.initial_conv(x)
        residual = self.residual_connection(x_initial)
        x_encoded = self.encoder(x_initial)
        x_out = x_encoded + residual
        return x_out


class ShortTermEncoder(nn.Module):
    def __init__(self, input_channels, out_z=64, ndf=16):
        super(ShortTermEncoder, self).__init__()
        self.ndf = ndf
        self.out_z = out_z

        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, ndf, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(ndf, ndf * 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, out_z, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_z),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.residual_connection = nn.Sequential(
            nn.Conv1d(ndf, out_z, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm1d(out_z)
        )

    def forward(self, x):
        x_initial = self.initial_conv(x)
        residual = self.residual_connection(x_initial)
        x_encoded = self.encoder(x_initial)
        x_out = x_encoded + residual
        return x_out


class LongTermDecoder(nn.Module):
    def __init__(self, output_channels, inout_seq_len_long, out_z=512, ndf=64):
        super(LongTermDecoder, self).__init__()
        self.ndf = ndf
        self.out_z = out_z
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(out_z, ndf * 8, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ndf * 8, ndf * 4, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ndf * 4, ndf * 2, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ndf * 2, ndf, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ndf, output_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.Tanh(),
            nn.Linear(16384, inout_seq_len_long)
        )

    def forward(self, x):
        return self.decoder(x)


class ShortTermDecoder(nn.Module):
    def __init__(self, output_channels, inout_seq_len_short, out_z=64, ndf=32):
        super(ShortTermDecoder, self).__init__()
        self.ndf = ndf
        self.out_z = out_z
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(out_z, output_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.Tanh(),
            nn.Linear(1024, inout_seq_len_short)
        )

    def forward(self, x):
        return self.decoder(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_embedding[:seq_len, :].unsqueeze(0)


def mask_input_signal_with_no_grad(input_signal, mask_ratio=0.15):
    batch_size, channels, seq_len = input_signal.shape
    device = input_signal.device
    mask = torch.zeros(batch_size, channels, seq_len,
                       dtype=torch.bool).to(device)
    mask_len = max(1, int(seq_len * mask_ratio))
    for b in range(batch_size):
        for c in range(channels):
            start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
            mask[b, c, start:start+mask_len] = True

    masked_signal = input_signal.clone()
    masked_signal[mask] = -100
    masked_signal = masked_signal.detach() * mask.logical_not() + \
        input_signal * mask.logical_not()

    return masked_signal, mask
