import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Linear(num_classes, hidden_size, bias=False)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, y):
        embeddings = self.embedding_table(y)
        return embeddings


#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################

def get_mask(batch, length, mask_ratio, device):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
    return: 
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {'mask': mask, 
            'ids_keep': ids_keep, 
            'ids_restore': ids_restore}


def mask_out_token(x, ids_keep):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token, extras=0):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + extras - x.shape[1], 1)
    x_ = torch.cat([x[:, extras:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :extras, :], x_], dim=1)  # append cls token
    return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, c_emb_dize, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_dize, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,  # 0 = unconditional
            learn_sigma=False,
            use_decoder=False,  # decide if add a lightweight DiT decoder
            mae_loss_coef=0,  # 0 = no mae loss
            pad_cls_token=False,  # decide if use cls_token as mask token for decoder
            direct_cls_token=False,  # decide if directly pass cls_toekn to decoder (0 = not pass to decoder)
            ext_feature_dim=0,  # decide if condition on external features (0 = no feature)
            use_encoder_feat=False,  # decide if condition on encoder output feature
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # normalize the encoder output feature
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.use_decoder = use_decoder
        self.mae_loss_coef = mae_loss_coef
        self.pad_cls_token = pad_cls_token
        self.direct_cls_token = direct_cls_token
        self.ext_feature_dim = ext_feature_dim
        self.use_encoder_feat = use_encoder_feat
        self.feat_norm = norm_layer(hidden_size, elementwise_affine=False)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) if num_classes else None
        num_patches = self.x_embedder.num_patches

        self.cls_token = None
        self.extras = 0
        self.decoder_extras = 0
        if self.pad_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.extras = 1
            self.decoder_extras = 1

        self.feat_embedder = None
        if self.ext_feature_dim > 0:
            self.feat_embedder = nn.Linear(self.ext_feature_dim, hidden_size, bias=True)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.extras, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.decoder_pos_embed = None
        self.decoder_layer = None
        self.decoder_blocks = None
        self.mask_token = None
        self.cls_token_embedder = None
        self.enc_feat_embedder = None
        final_hidden_size = hidden_size
        if self.use_decoder:
            decoder_hidden_size = 512
            decoder_depth = 8
            decoder_num_heads = 16
            if not self.direct_cls_token:
                self.decoder_extras = 0
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.decoder_extras, decoder_hidden_size),
                requires_grad=False)
            self.decoder_layer = DecoderLayer(hidden_size, decoder_hidden_size)
            self.decoder_blocks = nn.ModuleList([
                DiTBlock(decoder_hidden_size, hidden_size, decoder_num_heads, mlp_ratio=mlp_ratio) for _ in
                range(decoder_depth)
            ])
            if self.mae_loss_coef > 0:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))  # Similar to MAE
            if self.pad_cls_token:
                self.cls_token_embedder = nn.Linear(hidden_size, hidden_size, bias=True)
            if self.use_encoder_feat:
                self.enc_feat_embedder = nn.Linear(hidden_size, hidden_size, bias=True)
            final_hidden_size = decoder_hidden_size

        self.final_layer = FinalLayer(final_hidden_size, hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5),
                                            cls_token=self.pad_cls_token, extra_tokens=self.extras)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize cls_token embedding:
        if self.feat_embedder is not None:
            nn.init.normal_(self.feat_embedder.weight, std=0.02)

        # Initialize cls token
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=.02)

        # Initialize cls_token embedding:
        if self.cls_token_embedder is not None:
            nn.init.normal_(self.cls_token_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # --------------------------- decoder initialization ---------------------------
        # Initialize (and freeze) decoder_pos_embed by sin-cos embedding:
        if self.decoder_pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                int(self.x_embedder.num_patches ** 0.5),
                                                cls_token=self.pad_cls_token, extra_tokens=self.decoder_extras)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize mask token
        if self.mae_loss_coef > 0 and self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=.02)

        # Zero-out adaLN modulation layers in DiT decoder blocks:
        if self.decoder_blocks is not None:
            for block in self.decoder_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out decoder layers: (TODO: here we keep it the same with final layers but not sure if it makes sense)
        if self.decoder_layer is not None:
            nn.init.constant_(self.decoder_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.decoder_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.decoder_layer.linear.weight, 0)
            nn.init.constant_(self.decoder_layer.linear.bias, 0)
        # ------------------------------------------------------------------------------

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def encode(self, x, t, y, mask_ratio=0, mask_dict=None, feat=None):
        '''
        Encode x and (t, y, feat) into a latent representation.
        Return:
            x_feat: feature
            mask_dict with keys: 'ids_keep', 'ids_mask', 'mask_ratio'
        '''
        x = self.x_embedder(x) + self.pos_embed[:, self.extras:, :]  # (N, T, D), where T = H * W / patch_size ** 2
        if mask_ratio > 0 and mask_dict is None:
            mask_dict = get_mask(x.shape[0], x.shape[1], mask_ratio, device=x.device)
        if mask_ratio > 0:
            ids_keep = mask_dict['ids_keep']
            x = mask_out_token(x, ids_keep)
        # append cls token
        if self.cls_token is not None:
            cls_token = self.cls_token + self.pos_embed[:, :self.extras, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        t = self.t_embedder(t)  # (N, D)
        c = t
        if self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)
        assert (self.feat_embedder is None) or (self.enc_feat_embedder is None)
        if self.feat_embedder is not None:
            assert feat.shape[-1] == self.ext_feature_dim
            feat_embed = self.feat_embedder(feat)  # (N, D)
            c = c + feat_embed  # (N, D)
        if self.enc_feat_embedder is not None and feat is not None:
            assert feat.shape[-1] == c.shape[-1]
            feat_embed = self.enc_feat_embedder(feat)  # (N, D)
            c = c + feat_embed  # (N, D)

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)

        x_feat = x[:, self.extras:, :].mean(dim=1)  # global pool without cls token
        x_feat = self.feat_norm(x_feat)
        return x_feat, mask_dict


    def forward_encoder(self, x, t, y, mask_ratio=0, mask_dict=None, feat=None, train=True):
        '''
        Encode x and (t, y, feat) into a latent representation.
        Return:
            - out_enc: dict, containing the following keys: x, x_feat
            - c: the conditional embedding
        '''
        out_enc = dict()
        x = self.x_embedder(x) + self.pos_embed[:, self.extras:, :]  # (N, T, D), where T = H * W / patch_size ** 2
        if mask_ratio > 0 and mask_dict is None:
            mask_dict = get_mask(x.shape[0], x.shape[1], mask_ratio=mask_ratio, device=x.device)
        
        if mask_ratio > 0:
            ids_keep = mask_dict['ids_keep']
            ids_restore = mask_dict['ids_restore']
            if train:
                x = mask_out_token(x, ids_keep)

        # append cls token
        if self.cls_token is not None:
            cls_token = self.cls_token + self.pos_embed[:, :self.extras, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        t = self.t_embedder(t)  # (N, D)
        c = t
        if self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)
        assert (self.feat_embedder is None) or (self.enc_feat_embedder is None)
        if self.feat_embedder is not None:
            assert feat.shape[-1] == self.ext_feature_dim
            feat_embed = self.feat_embedder(feat)  # (N, D)
            c = c + feat_embed  # (N, D)
        if self.enc_feat_embedder is not None and feat is not None:
            assert feat.shape[-1] == c.shape[-1]
            feat_embed = self.enc_feat_embedder(feat)  # (N, D)
            c = c + feat_embed  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        out_enc['x'] = x

        return out_enc, c, mask_dict

    def forward(self, x, t, y, mask_ratio=0, mask_dict=None, feat=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if not self.training and self.use_encoder_feat:
            feat, _ = self.encode(x, t, y, feat=feat)
        out, c, mask_dict = self.forward_encoder(x, t, y, mask_ratio=mask_ratio, mask_dict=mask_dict, feat=feat, train=self.training)
        if mask_ratio > 0:
            ids_keep = mask_dict['ids_keep']
            ids_restore = mask_dict['ids_restore']
            out['mask'] = mask_dict['mask']
        else:
            ids_keep = ids_restore = None
        x = out['x']
        # Pass to a DiT decoder (if available)
        if self.use_decoder:
            if self.cls_token_embedder is not None:
                # cls_token_output = x[:, :self.extras, :].squeeze(dim=1).detach().clone()  # stop gradient
                cls_token_output = x[:, :self.extras, :].squeeze(dim=1)
                cls_token_embed = self.cls_token_embedder(self.feat_norm(cls_token_output))  # normalize cls token
                c = c + cls_token_embed  # pad cls_token output's embedding as feature conditioning

            assert self.decoder_layer is not None
            diff_extras = self.extras - self.decoder_extras
            x = self.decoder_layer(x[:, diff_extras:, :], c)  # remove cls token (if necessary)
            if self.training and mask_ratio > 0:
                mask_token = self.mask_token
                if mask_token is None:
                    mask_token = torch.zeros(1, 1, x.shape[2]).to(x)  # concat zeros to match shape
                x = unmask_tokens(x, ids_restore, mask_token, extras=self.decoder_extras)
            assert self.decoder_pos_embed is not None
            x = x + self.decoder_pos_embed
            assert self.decoder_blocks is not None
            for block in self.decoder_blocks:
                x = block(x, c)  # (N, T, D)

        x = self.final_layer(x, c)  # (N, T or T+1, patch_size ** 2 * out_channels)
        if not self.use_decoder and (self.training and mask_ratio > 0):
            mask_token = torch.zeros(1, 1, x.shape[2]).to(x)  # concat zeros to match shape
            x = unmask_tokens(x, ids_restore, mask_token, extras=self.extras)
        x = x[:, self.decoder_extras:, :]  # remove cls token (if necessary)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        out['x'] = x
        return out

    def forward_with_cfg(self, x, t, y, cfg_scale, feat=None, **model_kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        out = dict()

        # Setup classifier-free guidance
        x = torch.cat([x, x], 0)
        y_null = torch.zeros_like(y)
        y = torch.cat([y, y_null], 0)
        if feat is not None:
            feat = torch.cat([feat, feat], 0)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        assert self.num_classes and y is not None
        model_out = self.forward(combined, t, y, feat=feat)['x']
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        half_rest = rest[: len(rest) // 2]
        x = torch.cat([half_eps, half_rest], dim=1)
        out['x'] = x
        return out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_H_2(**kwargs):
    return DiT(depth=32, hidden_size=1280, patch_size=2, num_heads=16, **kwargs)


def DiT_H_4(**kwargs):
    return DiT(depth=32, hidden_size=1280, patch_size=4, num_heads=16, **kwargs)


def DiT_H_8(**kwargs):
    return DiT(depth=32, hidden_size=1280, patch_size=8, num_heads=16, **kwargs)


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-H/2': DiT_H_2, 'DiT-H/4': DiT_H_4, 'DiT-H/8': DiT_H_8,
    'DiT-XL/2': DiT_XL_2, 'DiT-XL/4': DiT_XL_4, 'DiT-XL/8': DiT_XL_8,
    'DiT-L/2': DiT_L_2, 'DiT-L/4': DiT_L_4, 'DiT-L/8': DiT_L_8,
    'DiT-B/2': DiT_B_2, 'DiT-B/4': DiT_B_4, 'DiT-B/8': DiT_B_8,
    'DiT-S/2': DiT_S_2, 'DiT-S/4': DiT_S_4, 'DiT-S/8': DiT_S_8,
}


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPPrecond(nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution.
                 img_channels,  # Number of color channels.
                 num_classes=0,  # Number of class labels, 0 = unconditional.
                 beta_d=19.9,  # Extent of the noise level schedule.
                 beta_min=0.1,  # Initial slope of the noise level schedule.
                 M=1000,  # Original number of timesteps in the DDPM formulation.
                 epsilon_t=1e-5,  # Minimum t-value used during training.
                 model_type='DiT-B/2',  # Class name of the underlying model.
                 **model_kwargs,  # Keyword arguments for the underlying model.
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = DiT_models[model_type](input_size=img_resolution, in_channels=img_channels,
                                            num_classes=num_classes, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, cfg_scale=None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        model_out = model_fn((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, **model_kwargs)
        F_x = model_out['x']
        D_x = c_skip * x + c_out * F_x
        model_out['x'] = D_x
        return model_out

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VEPrecond(nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution.
                 img_channels,  # Number of color channels.
                 num_classes=0,  # Number of class labels, 0 = unconditional.
                 sigma_min=0.02,  # Minimum supported noise level.
                 sigma_max=100,  # Maximum supported noise level.
                 model_type='DiT-B/2',  # Class name of the underlying model.
                 **model_kwargs,  # Keyword arguments for the underlying model.
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = DiT_models[model_type](input_size=img_resolution, in_channels=img_channels,
                                            num_classes=num_classes, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, cfg_scale=None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        model_out = model_fn((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, **model_kwargs)
        F_x = model_out['x']
        D_x = c_skip * x + c_out * F_x
        model_out['x'] = D_x
        return model_out

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

class iDDPMPrecond(nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution.
                 img_channels,  # Number of color channels.
                 num_classes=0,  # Number of class labels, 0 = unconditional.
                 C_1=0.001,  # Timestep adjustment at low noise levels.
                 C_2=0.008,  # Timestep adjustment at high noise levels.
                 M=1000,  # Original number of timesteps in the DDPM formulation.
                 model_type='DiT-B/2',  # Class name of the underlying model.
                 **model_kwargs,  # Keyword arguments for the underlying model.
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = DiT_models[model_type](input_size=img_resolution, in_channels=img_channels,
                                            num_classes=num_classes, learn_sigma=True, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, cfg_scale=None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(x.dtype)

        model_out = model_fn((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, **model_kwargs)
        F_x = model_out['x']
        D_x = c_skip * x + c_out * F_x
        model_out['x'] = D_x
        return model_out

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(sigma.dtype).reshape(1, -1, 1),
                            self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond(nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution.
                 img_channels,  # Number of color channels.
                 num_classes=0,  # Number of class labels, 0 = unconditional.
                 sigma_min=0,  # Minimum supported noise level.
                 sigma_max=float('inf'),  # Maximum supported noise level.
                 sigma_data=0.5,  # Expected standard deviation of the training data.
                 model_type='DiT-B/2',  # Class name of the underlying model.
                 **model_kwargs,  # Keyword arguments for the underlying model.
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = DiT_models[model_type](input_size=img_resolution, in_channels=img_channels,
                                            num_classes=num_classes, **model_kwargs)

    def encode(self, x, sigma, class_labels=None, **model_kwargs):

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        feat, mask_dict = self.model.encode((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, **model_kwargs)
        return feat

    def forward(self, x, sigma, class_labels=None, cfg_scale=None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        model_out = model_fn((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, **model_kwargs)
        F_x = model_out['x']
        D_x = c_skip * x + c_out * F_x
        model_out['x'] = D_x
        return model_out

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


Precond_models = {
    'vp': VPPrecond, 've': VEPrecond, 'edm': EDMPrecond
}
