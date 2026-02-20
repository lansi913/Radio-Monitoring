import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMoEBlock(nn.Module):
    def __init__(self,
                 token_dim,
                 num_heads=8,
                 num_attention_layers=3,
                 num_experts=4,
                 num_tokens=10,
                 top_k_long=5,
                 top_k_short=3,
                 use_batch_norm=True,
                 use_residual=True):
        super(AttentionMoEBlock, self).__init__()

        self.token_dim = token_dim
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.top_k_long = top_k_long
        self.top_k_short = top_k_short

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=num_heads,
                batch_first=True
            )
            for _ in range(num_attention_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(token_dim)
            for _ in range(num_attention_layers)
        ])

        self.long_moe = MoE(token_dim, token_dim, num_experts, num_tokens)
        self.short_moe = MoE(token_dim, token_dim, num_experts, num_tokens)

        if use_batch_norm:
            self.long_topk_bn = nn.BatchNorm1d(top_k_long)
            self.short_topk_bn = nn.BatchNorm1d(top_k_short)
        else:
            self.long_topk_bn = nn.Identity()
            self.short_topk_bn = nn.Identity()

        self.output_projection = nn.Linear(top_k_long + top_k_short, token_dim)
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        if isinstance(x, tuple):
            residual = x[0].clone() if self.use_residual else None
            x = x[0]
        else:
            residual = x.clone() if self.use_residual else None

        attn_output = x
        for i, (attn_layer, norm_layer) in enumerate(zip(self.attention_layers, self.attention_norms)):
            attn_output, _ = attn_layer(attn_output, attn_output, attn_output)
            attn_output = norm_layer(attn_output)

        batch_size, seq_len, _ = attn_output.shape
        moe_input = attn_output.reshape(-1, self.token_dim)

        mo_output_long = self.long_moe(moe_input)
        top_k_expert_long = torch.topk(
            mo_output_long, self.top_k_long, dim=1).values
        top_k_expert_long = self.long_topk_bn(top_k_expert_long)

        moe_output_short = self.short_moe(moe_input)
        top_k_expert_short = torch.topk(
            moe_output_short, self.top_k_short, dim=1).values
        top_k_expert_short = self.short_topk_bn(top_k_expert_short)

        combined_moe = torch.cat(
            [top_k_expert_long, top_k_expert_short], dim=1)
        projected_output = self.output_projection(combined_moe)
        output = projected_output.reshape(batch_size, seq_len, self.token_dim)

        if self.use_residual and residual is not None:
            output = output + residual

        output = self.final_norm(output)

        intermediate_results = {
            'attention_output': attn_output,
            'moe_long': mo_output_long.reshape(batch_size, seq_len, -1),
            'moe_short': moe_output_short.reshape(batch_size, seq_len, -1),
            'top_k_long': top_k_expert_long.reshape(batch_size, seq_len, -1),
            'top_k_short': top_k_expert_short.reshape(batch_size, seq_len, -1)
        }

        return output, intermediate_results


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_tokens):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.num_tokens = num_tokens

        self.shared_ffn = nn.Linear(input_dim, hidden_dim)
        self.expert_ffns = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])

        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        shared_output = self.shared_ffn(x)
        expert_scores = F.softmax(self.router(x), dim=-1)
        token_expert_map = torch.argmax(expert_scores, dim=-1)

        routed_outputs = torch.zeros_like(shared_output)
        for i in range(self.num_experts):
            mask = (token_expert_map == i).float().unsqueeze(-1)
            routed_outputs += mask * self.expert_ffns[i](x)

        output = shared_output + routed_outputs
        return output
