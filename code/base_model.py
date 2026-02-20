import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.Bone import LongTermEncoder, ShortTermEncoder, LongTermDecoder, ShortTermDecoder, LearnablePositionalEncoding
from Module.Router import AttentionMoEBlock, MoE


class RadarPretrainingModel(nn.Module):
    def __init__(self, input_dim_long, input_dim_short, inout_seq_len_long, inout_seq_len_short, hidden_dim, token_dim, num_heads, num_tokens, num_experts, top_k_long, top_k_short):
        super(RadarPretrainingModel, self).__init__()

        self.long_postion_encoder = LearnablePositionalEncoding(
            inout_seq_len_long, input_dim_long)
        self.short_postion_encoder = LearnablePositionalEncoding(
            input_dim_short, inout_seq_len_short)

        self.long_conv_encoder = LongTermEncoder(input_dim_long, hidden_dim)
        self.short_conv_encoder = ShortTermEncoder(input_dim_short, hidden_dim)

        self.long_bn = nn.BatchNorm1d(hidden_dim)
        self.short_bn = nn.BatchNorm1d(hidden_dim)

        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.liear_embeb = nn.Linear(hidden_dim, token_dim)

        self.attention_moe_block_1 = AttentionMoEBlock(
            token_dim=512,
            num_heads=32,
            num_attention_layers=3,
            num_experts=32,
            num_tokens=30,
            top_k_long=15,
            top_k_short=12,
            use_batch_norm=True,
            use_residual=True
        )

        self.attention_moe_block_2 = AttentionMoEBlock(
            token_dim=512,
            num_heads=16,
            num_attention_layers=3,
            num_experts=4,
            num_tokens=10,
            top_k_long=10,
            top_k_short=8,
            use_batch_norm=True,
            use_residual=True
        )

        self.attention_moe_block_3 = AttentionMoEBlock(
            token_dim=512,
            num_heads=8,
            num_attention_layers=3,
            num_experts=4,
            num_tokens=10,
            top_k_long=5,
            top_k_short=3,
            use_batch_norm=True,
            use_residual=True
        )

        self.long_moe = MoE(token_dim, token_dim, num_experts, num_tokens)
        self.short_moe = MoE(token_dim, token_dim, num_experts, num_tokens)

        self.top_k_long = top_k_long
        self.top_k_short = top_k_short
        self.long_topk_bn = nn.BatchNorm1d(top_k_long)
        self.short_topk_bn = nn.BatchNorm1d(top_k_short)

        self.long_term_decoder = LongTermDecoder(
            input_dim_long, inout_seq_len_long, top_k_long)
        self.short_term_decoder = ShortTermDecoder(
            input_dim_short, inout_seq_len_short, top_k_short)
        self.inout_seq_len_long = inout_seq_len_long
        self.inout_seq_len_short = inout_seq_len_short
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim

    def forward(self, x_long=None, x_short=None, batch_size=None, input_ids=None, **kwargs):
        if input_ids is not None:
            if isinstance(input_ids, (tuple, list)) and len(input_ids) == 3:
                x_long, x_short, batch_size = input_ids
            else:
                raise ValueError(
                    "input_ids must be a tuple or list of (x_long, x_short, batch_size)")

        long_postion_embedding = self.long_postion_encoder(x_long)
        short_postion_embedding = self.short_postion_encoder(x_short)

        long_encoded = self.long_conv_encoder(long_postion_embedding)
        short_encoded = self.short_conv_encoder(short_postion_embedding)

        long_encoded = self.long_bn(long_encoded)
        short_encoded = self.short_bn(short_encoded)

        long_latent = long_encoded.reshape(batch_size, -1)
        short_latent = short_encoded.reshape(batch_size, -1)

        combined_encoded = torch.cat([long_latent, short_latent], dim=1)
        combined_encoded = combined_encoded.reshape(
            batch_size, -1, self.hidden_dim)
        imbedding_dim = self.liear_embeb(combined_encoded)

        attn_moe_output = self.attention_moe_block_1(imbedding_dim)
        attn_moe_output = self.attention_moe_block_2(attn_moe_output)
        attn_moe_output = self.attention_moe_block_3(attn_moe_output)

        moe_output_long = self.long_moe(attn_moe_output[0])
        top_k_expert_long = torch.topk(
            moe_output_long, self.top_k_long, dim=1).values

        moe_output_short = self.short_moe(attn_moe_output[0])
        top_k_expert_short = torch.topk(
            moe_output_short, self.top_k_short, dim=1).values

        top_k_expert_long = self.long_topk_bn(top_k_expert_long)
        top_k_expert_short = self.short_topk_bn(top_k_expert_short)

        long_decoded = self.long_term_decoder(top_k_expert_long)
        short_decoded = self.short_term_decoder(top_k_expert_short)

        return long_decoded, short_decoded, moe_output_long, moe_output_short
