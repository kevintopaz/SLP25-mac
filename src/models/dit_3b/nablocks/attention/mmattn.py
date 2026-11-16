# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from .....common.cache import Cache
from .....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv
from .....common.half_precision_fixes import safe_pad_operation

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import get_window_op
from itertools import chain


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)

        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        # Attention handles dtype conversion internally using pipeline compute_dtype
        attn = self.attn(
            q=concat(vid_q, txt_q),
            k=concat(vid_k, txt_k),
            v=concat(vid_v, txt_v),
            cu_seqlens_q=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache("mm_maxlen", lambda: all_len.max()),
            max_seqlen_k=cache("mm_maxlen", lambda: all_len.max()),
        ).type_as(vid_q)

        attn = rearrange(attn, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(*args, attention_mode=attention_mode, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)
        # Optional temporal window override (latent frames); None = computed from window.
        self.fixed_wt = None

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        if _window_groups_wanted(vid_shape):
            return self._forward_window_groups(vid, txt, vid_shape, txt_shape, cache)
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_{self.fixed_wt}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window, fixed_wt=self.fixed_wt)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        # window rope
        if self.rope:
            if self.rope.mm:
                # repeat text q and k for window mmrope
                _, num_h, _ = txt_q.shape
                txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
                txt_q_repeat = na.unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [[x] * n for x, n in zip(txt_q_repeat, window_count)]
                txt_q_repeat = list(chain(*txt_q_repeat))
                txt_q_repeat, txt_shape_repeat = na.flatten(txt_q_repeat)
                txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)

                txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
                txt_k_repeat = na.unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [[x] * n for x, n in zip(txt_k_repeat, window_count)]
                txt_k_repeat = list(chain(*txt_k_repeat))
                txt_k_repeat, _ = na.flatten(txt_k_repeat)
                txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, window_shape, txt_q_repeat, txt_k_repeat, txt_shape_repeat, cache_win
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)
            
        # Attention handles dtype conversion internally using pipeline compute_dtype
        out = self.attn(
            q=concat_win(vid_q, txt_q),
            k=concat_win(vid_k, txt_k),
            v=concat_win(vid_v, txt_v),
            cu_seqlens_q=cache_win(
                "vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            cu_seqlens_k=cache_win(
                "vid_seqlens_k", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max()),
            max_seqlen_k=cache_win("vid_max_seqlen_k", lambda: all_len_win.max()),
        ).type_as(vid_q)

        # text pooling
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out


# ── window-group loop (large inputs) ─────────────────────────────────────────
# Windows never interact inside a block, so a block's attention can run a few
# windows at a time: each group gathers its tokens, projects QKV, normalizes,
# applies RoPE and attends -- the full-sequence QKV / RoPE / attention tensors
# (~4 GB each at 121 frames 1080p) never exist. Text outputs are the mean over
# all windows (as unconcat_coalesce): group means weighted by window count,
# accumulated in float32. Same math as the full pass (summation order aside);
# the scheme the MLX DiT uses on Apple silicon.
import os as _os


def _window_groups_wanted(vid_shape: torch.LongTensor) -> bool:
    lim = int(_os.environ.get("DIT_WINDOW_GROUP_TOKENS", "16384"))
    if lim <= 0 or vid_shape.shape[0] != 1 or int(vid_shape.prod(-1).sum()) <= lim:
        return False
    try:
        from .....common.distributed.advanced import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size() == 1
    except Exception:
        return True


def _forward_window_groups(self, vid, txt, vid_shape, txt_shape, cache):
    lim = int(_os.environ.get("DIT_WINDOW_GROUP_TOKENS", "16384"))
    cache_win = cache.namespace(f"{self.window_method}_{self.window}_{self.fixed_wt}_sd3")

    def make_window(x: torch.Tensor):
        t, h, w, _ = x.shape
        window_slices = self.window_op((t, h, w), self.window, fixed_wt=self.fixed_wt)
        return [x[st, sh, sw] for (st, sh, sw) in window_slices]

    window_partition, window_reverse, window_shape, window_count = cache_win(
        "win_transform", lambda: na.window_idx(vid_shape, make_window))
    perm = cache_win("win_perm", lambda: window_partition(
        torch.arange(int(vid_shape.prod(-1).sum()), device=vid.device).unsqueeze(-1)).squeeze(-1))
    win_len = cache_win("vid_len", lambda: window_shape.prod(-1))
    n_win = int(window_shape.shape[0])
    lens = win_len.tolist()
    groups, start, acc = [], 0, 0                         # consecutive windows, <= lim tokens each
    for i, l in enumerate(lens):
        if acc and acc + l > lim:
            groups.append((start, i))
            start, acc = i, 0
        acc += l
    groups.append((start, n_win))
    offs = [0]
    for l in lens:
        offs.append(offs[-1] + l)

    txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
    vid_out = None
    txt_sum = None
    for gi, (w0, w1) in enumerate(groups):
        ng = w1 - w0
        idx = perm[offs[w0]:offs[w1]]
        gshape = window_shape[w0:w1]
        cache_g = cache_win.namespace(f"grp{w0}_{w1}")
        vid_qkv, txt_qkv = self.proj_qkv(vid.index_select(0, idx), txt)
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)
        if self.rope:
            if self.rope.mm:
                txt_shape_rep = txt_shape.repeat(ng, 1)
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, gshape, txt_q.repeat(ng, 1, 1), txt_k.repeat(ng, 1, 1), txt_shape_rep, cache_g)
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, gshape, cache_g)
        vlen = win_len[w0:w1]
        alen = vlen + txt_len.repeat_interleave(ng)
        concat_g, unconcat_g = cache_g("mm_pnp", lambda: na.repeat_concat_idx(vlen, txt_len, ng))
        cu = cache_g("seqlens", lambda: safe_pad_operation(alen.cumsum(0), (1, 0)).int())
        mx = cache_g("max_seqlen", lambda: alen.max())
        out = self.attn(q=concat_g(vid_q, txt_q), k=concat_g(vid_k, txt_k), v=concat_g(vid_v, txt_v),
                        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx).type_as(vid_q)
        vo, to = unconcat_g(out)
        del out, vid_q, vid_k, vid_v, vid_qkv
        vo = rearrange(vo, "l h d -> l (h d)")
        if vid_out is None:
            vid_out = vo.new_empty((vid.shape[0], vo.shape[1]))
        vid_out.index_copy_(0, idx, vo)                    # back to the token order (window_reverse)
        to = rearrange(to, "l h d -> l (h d)").float() * ng
        txt_sum = to if txt_sum is None else txt_sum + to
        del vo, to
    txt_out = (txt_sum / n_win).to(vid_out.dtype)
    return self.proj_out(vid_out, txt_out)


NaSwinAttention._forward_window_groups = _forward_window_groups
