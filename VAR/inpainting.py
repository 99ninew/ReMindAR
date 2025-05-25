def autoregressive_infer_cfg_with_mask(
    self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
    more_smooth=False,
    input_img_tokens: Optional[List[torch.Tensor]] = None, edit_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
    """
    only used for inference, on autoregressive mode
    :param B: batch size
    :param label_B: imagenet label; if None, randomly sampled
    :param g_seed: random seed
    :param cfg: classifier-free guidance ratio
    :param top_k: top-k sampling
    :param top_p: top-p sampling
    :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
    :param input_img_tokens: (optional, only for zero-shot edit tasks) tokens of the image to be edited
    :param edit_mask: (optional, only for zero-shot edit tasks) binary mask, 1 for keeping given tokens; 0 for generating new tokens
    :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
    """
    if g_seed is None: rng = None
    else: self.rng.manual_seed(g_seed); rng = self.rng

    if label_B is None:
        label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

    sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

    lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

    for b in self.blocks: b.attn.kv_caching(True)
    for si, pn in enumerate(self.patch_nums):   # si: i-th segment
        ratio = si / self.num_stages_minus_1
        # last_L = cur_L
        cur_L += pn*pn
        # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map
        AdaLNSelfAttn.forward
        for b in self.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = self.get_logits(x, cond_BD)

        t = cfg * ratio
        logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        if not more_smooth: # this is the default case
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
        else:   # not used when evaluating FID/IS/Precision/Recall
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
        if edit_mask is not None:
            gt_BChw = self.vae_quant_proxy[0].embedding(input_img_tokens[si]).transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            h_BChw = replace_embedding(edit_mask, h_BChw, gt_BChw, pn, pn)

        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
        if si != self.num_stages_minus_1:   # prepare for next stage
            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

    for b in self.blocks: b.attn.kv_caching(False)
    return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)


def replace_embedding(edit_mask: torch.Tensor, h_BChw: torch.Tensor, gt_BChw: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    B = h_BChw.shape[0]
    h, w = edit_mask.shape[-2:]
    if edit_mask.ndim == 2:
        edit_mask = edit_mask.unsqueeze(0).expand(B, h, w)

    force_gt_B1hw = F.interpolate(edit_mask.unsqueeze(1).to(dtype=torch.float, device=gt_BChw.device), size=(ph, pw), mode='bilinear', align_corners=False).gt(0.5).int()
    if ph * pw <= 3: force_gt_B1hw.fill_(1)
    return gt_BChw * force_gt_B1hw + h_BChw * (1 - force_gt_B1hw)
