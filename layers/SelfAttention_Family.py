import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class RWKVAttention(nn.Module):
    """
    RWKV / State-Space style causal attention (simplified).

    This implementation keeps the same API as other Attention classes in this file.
    It applies an additive positional decay factor alpha^{t-s} (implemented via
    adding (t-s)*log_alpha to the pre-softmax scores) so that more recent keys
    receive higher weight. The decay parameter is learnable per-head.

    Complexity: O(L^2) memory/time (naive), but it offers a simple replacement
    that captures the RWKV-style temporal decay behaviour while keeping API
    compatibility. For very long sequences you may want to replace with a
    linear-time recurrence implementation.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(RWKVAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # learnable log-alpha per head (initialized near 0 -> alpha~1)
        # actual head count is not known at init (AttentionLayer sets heads),
        # so we lazily create when seen first forward call.
        self._log_alpha = None

    def _init_log_alpha(self, heads, device, dtype):
        # initialize log_alpha to a small negative value so alpha slightly < 1
        self._log_alpha = nn.Parameter(torch.full((heads,), -0.1, device=device, dtype=dtype))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # queries/keys/values shapes: queries (B, L, H, E), values (B, S, H, D)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if self._log_alpha is None:
            # lazy init based on encountered number of heads
            self._init_log_alpha(H, queries.device, queries.dtype)

        scale = self.scale or 1. / sqrt(E)

        # base similarity matrix (same as FullAttention)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # optional de-stationary rescaling like DSAttention
        if tau is not None or delta is not None:
            tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
            delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
            scores = scores * tau + delta

        # add RWKV-style causal decay: add (t-s) * log_alpha_h to scores
        # build relative index matrix (L, S) where rel[t,s] = t - s
        rel = (torch.arange(L, device=queries.device).unsqueeze(1) - torch.arange(S, device=queries.device).unsqueeze(0)).float()
        # clamp negatives since we only want causal (s<=t) contributions; we'll mask later
        # rel shape -> (L, S) -> expand to (1, H, L, S)
        rel = rel.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        log_alpha = self._log_alpha.view(1, H, 1, 1)  # (1,H,1,1)
        decay_term = rel * log_alpha  # broadcast -> (1,H,L,S)
        # decay_term might be negative for s>t (rel<0); the mask will remove them
        scores = scores + decay_term

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            neg_inf = torch.finfo(scores.dtype).min
            scores.masked_fill_(attn_mask.mask, neg_inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out


class MultiScaleAttention(nn.Module):
    """Multi-scale self/cross attention.

    If queries and keys have equal length (self-attention) it computes attention on
    multiple downsampled resolutions and fuses the upsampled results with learned
    weights. For cross-attention (length differs) it falls back to a single full
    attention computation.
    """
    def __init__(self, mask_flag=True, scales=(1, 2, 4), attention_dropout=0.1, output_attention=False, fuse="softmax"):
        super(MultiScaleAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scales = tuple(sorted(set([s for s in scales if isinstance(s, int) and s >= 1])))
        if len(self.scales) == 0:
            self.scales = (1,)
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.fuse = fuse
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        B, L, H, D = x.shape
        if L % factor != 0:
            return None
        x = x.view(B, L // factor, factor, H, D).mean(dim=2)
        return x

    def _upsample(self, x_small, factor, L_full):
        if factor == 1:
            return x_small
        B, Ls, H, D = x_small.shape
        # repeat each token factor times
        x = x_small.unsqueeze(2).repeat(1, 1, factor, 1, 1).view(B, Ls * factor, H, D)
        if x.shape[1] != L_full:
            # safety crop/pad
            if x.shape[1] > L_full:
                x = x[:, :L_full]
            else:
                pad_len = L_full - x.shape[1]
                pad = torch.zeros(B, pad_len, H, D, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        return x

    def _single_attention(self, q, k, v, attn_mask):
        B, Lq, H, E = q.shape
        _, Lk, _, D = v.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        if self.mask_flag and Lq == Lk:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, Lq, device=q.device)
            scores.masked_fill_(attn_mask.mask, torch.finfo(scores.dtype).min)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", A, v)
        return out, A if self.output_attention else None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, Lq, H, E = queries.shape
        _, Lk, _, D = values.shape
        # fallback to single-scale if not self-attention
        if Lq != Lk:
            out, attn = self._single_attention(queries, keys, values, attn_mask)
            return out.contiguous(), attn
        L = Lq
        valid_scales = [s for s in self.scales if L % s == 0]
        if len(valid_scales) == 0:
            valid_scales = [1]
        multi_outputs = []
        attn_return = None
        for idx, s in enumerate(valid_scales):
            q_ds = self._downsample(queries, s)
            k_ds = self._downsample(keys, s)
            v_ds = self._downsample(values, s)
            if q_ds is None or k_ds is None or v_ds is None:
                continue
            out_s, attn_s = self._single_attention(q_ds, k_ds, v_ds, attn_mask=None)
            out_up = self._upsample(out_s, s, L)
            multi_outputs.append(out_up)
            if attn_return is None and attn_s is not None:
                # upsample attention for visualization (optional simplistic repeat)
                if self.output_attention:
                    attn_up = attn_s.unsqueeze(2).repeat(1, 1, s, 1, 1).view(B, H, attn_s.shape[2] * s, attn_s.shape[3] * s)
                    attn_return = attn_up
        if len(multi_outputs) == 0:
            out, attn = self._single_attention(queries, keys, values, attn_mask)
            return out.contiguous(), attn
        stacked = torch.stack(multi_outputs, dim=0)  # [SCALE, B, L, H, D]
        logits = self.scale_logits[: stacked.shape[0]]
        if self.fuse == "softmax":
            weights = torch.softmax(logits, dim=0).view(-1, 1, 1, 1, 1)
        else:
            weights = torch.sigmoid(logits).view(-1, 1, 1, 1, 1)
            weights = weights / weights.sum()
        fused = (weights * stacked).sum(dim=0)
        return fused.contiguous(), attn_return
    
class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(TSMixer, self).__init__()

        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q_proj(q).reshape(B, L, H, -1)
        k = self.k_proj(k).reshape(B, S, H, -1)
        v = self.v_proj(v).reshape(B, S, H, -1)

        out, attn = self.attention(
            q, k, v,
            res=res, attn=attn
        )
        out = out.view(B, L, -1)

        return self.out(out), attn


class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None, attn_map=False, nst=False):
        super(ResAttention, self).__init__()

        self.nst = nst
        self.scale = scale
        self.attn_map = attn_map
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        if self.attn_map is True:
            heat_map = attn_map.reshape(32, -1, H, L, S)
            for b in range(heat_map.shape[0]):
                for c in range(heat_map.shape[1]):
                    h_map = heat_map[b, c, 0, ...].detach().cpu().numpy()
                    # plt.savefig(heat_map, f'{b} sample {c} channel')

                    plt.figure(figsize=(10, 8), dpi=200)
                    plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                    plt.colorbar()

                    # 设置X轴和Y轴的标签为黑体文字
                    plt.rcParams['font.family'] = 'serif'
                    plt.rcParams['font.serif'] = ['Times New Roman']
                    plt.xlabel('Key Time Patch', fontsize=14)
                    plt.ylabel('Query Time Patch', fontsize=14)
                    plt.tight_layout()
                    if self.nst is True:
                        plt.savefig(f'./time map/{b}_sample_{c}_channel.png')
                    else:
                        plt.savefig(f'./stable time map/{b}_sample_{c}_channel.png')
                    # 关闭当前图形窗口
                    plt.close()
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A

