import math
from functools import partial
import random 

import torch
import torch.nn.functional as F
from torch import nn, einsum

import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat


from .t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from tqdm.auto import tqdm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            context_mask = F.pad(context_mask, (1, 0), value = True)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, context = None, context_mask = None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context = context, context_mask = context_mask) + x

            x = ff(x) + x

        return self.norm(x)



class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out = None,
        t5_name = DEFAULT_T5_NAME,
        self_cond = False,
        add_mask_id = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim = dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias = False)

        # text conditioning

        self.encode_text = partial(t5_encode_text, name = t5_name)

        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        neg_logits = self.forward(*args, neg_text_embed = neg_text_embed, cond_drop_prob = 0., **kwargs)
        pos_logits, embed = self.forward(*args, return_embed = True, text_embed = text_embed, cond_drop_prob = 0., **kwargs)

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        return_embed = False,
        return_logits = False,
        labels = None,
        ignore_index = 0,
        self_cond_embed = None,
        cond_drop_prob = 0.,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        # prepare texts

        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            text_embeds = self.encode_text(texts)

        context = self.text_embed_proj(text_embeds)

        context_mask = (text_embeds != 0).any(dim = -1)

        # classifier free guidance

        if self.training and cond_drop_prob > 0.:
            mask = prob_mask_like((b, 1), 1. - cond_drop_prob, device)
            context_mask = context_mask & mask

        # concat conditioning image token ids if needed

        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(conditioning_token_ids, 'b ... -> b (...)')
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim = -2)
            context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value = True)

        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context = context, context_mask = context_mask)

        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = ignore_index)

        if not return_logits:
            return loss

        return loss, logits




class MusicTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        n_q=8, # quantification dim for music
        dim_out = None,
        t5_name = DEFAULT_T5_NAME,
        self_cond = False,
        add_mask_id = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        
        # self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim) 
        # token embedding for different codec
        self.token_emb = nn.ModuleList([nn.Embedding(num_tokens + int(add_mask_id), dim) for _ in range(n_q)])
        self.pos_emb = nn.Embedding(seq_len*2, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim = dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias = False) # not consider mask token

        # text conditioning
        # always re-load make time consuming
        self.encode_text = partial(t5_encode_text, name = t5_name)

        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        neg_logits = self.forward(*args, neg_text_embed = neg_text_embed, cond_drop_prob = 0., **kwargs)
        pos_logits, embed = self.forward(*args, return_embed = True, text_embed = text_embed, cond_drop_prob = 0., **kwargs)

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        idx, # idx-th row for music code matrix 
        return_embed = False,
        return_logits = False,
        labels = None,
        ignore_index = 0,
        self_cond_embed = None,
        cond_drop_prob = 0.,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
    ):
        device, b, n_q, n = x.device, *x.shape
        assert n <= self.seq_len
        assert n_q == self.n_q 
        assert idx < self.n_q 
        # prepare texts

        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            text_embeds = self.encode_text(texts)

        context = self.text_embed_proj(text_embeds)

        context_mask = (text_embeds != 0).any(dim = -1)

        # classifier free guidance

        if self.training and cond_drop_prob > 0.:
            mask = prob_mask_like((b, 1), 1. - cond_drop_prob, device)
            context_mask = context_mask & mask

        # sum up history music tokens 
        if idx == 0:
            history_ = torch.zeros((b, n, self.dim)).to(device)
        else:
            history_ = [self.token_emb[k](x[:, k]) for k in range(idx)]
            history_  = torch.stack(history_, dim=0) 
            history_ = torch.sum(history_, dim=0)
        
        x = self.token_emb[idx](x[:, idx]) 
        x = torch.cat((history_, x), dim=1) 
        x = x + self.pos_emb(torch.arange(x.size(1), device = device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context = context, context_mask = context_mask)

        # select the last n hidden states 
        embed = embed[:, -n:, :] 
        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = ignore_index)

        if not return_logits:
            return loss

        return loss, logits

# self critic wrapper

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed = True, **kwargs)
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed = True, **kwargs)
        return self.to_pred(embeds)

    def forward(self, x, *args, labels = None, **kwargs):
        _, embeds = self.net(x, *args, return_embed = True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        logits = rearrange(logits, '... 1 -> ...')
        return F.binary_cross_entropy_with_logits(logits, labels)

# specialized transformers

class MaskGmtTransformer(MusicTransformer):
    def __init__(self, *args, **kwargs):
        assert 'add_mask_id' not in kwargs
        super().__init__(*args, add_mask_id = True, **kwargs)

class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert 'dim_out' not in kwargs
        super().__init__(*args, dim_out = 1, **kwargs)

# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device = None):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# noise schedules

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

# main maskgit classes


class MaskGmt(nn.Module):
    def __init__(
        self,
        transformer: MaskGmtTransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic = False,
        cond_drop_prob = 0.5,
        self_cond_prob = 0.9,
        no_mask_token_prob = 0.,
        critic_loss_weight = 1.
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond
        
        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts: List[str],
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size = None,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        force_not_use_token_critic = False,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        cond_scale = 3,
        critic_noise_scale = 1
    ):
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))

        # begin with all image token ids masked

        device = next(self.parameters()).device

        seq_len = fmap_size ** 2

        batch_size = len(texts)

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_id, dtype = torch.long, device = device)
        scores = torch.zeros(shape, dtype = torch.float32, device = device)

        starting_temperature = temperature

        cond_ids = None

        text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        neg_text_embeds = None
        if exists(negative_texts):
            assert len(texts) == len(negative_texts)

            neg_text_embeds = self.transformer.encode_text(negative_texts)
            demask_fn = partial(self.transformer.forward_with_neg_prompt, neg_text_embeds = neg_text_embeds)

            if use_token_critic:
                token_critic_fn = partial(self.token_critic.forward_with_neg_prompt, neg_text_embeds = neg_text_embeds)

        if self.resize_image_for_cond_image:
            assert exists(cond_images), 'conditioning image must be passed in to generate for super res maskgit'
            with torch.no_grad():
                _, cond_ids, _ = self.cond_vae.encode(cond_images)

        self_cond_embed = None

        for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim = -1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            logits, embed = demask_fn(
                ids,
                text_embeds = text_embeds,
                self_cond_embed = self_cond_embed,
                conditioning_token_ids = cond_ids,
                cond_scale = cond_scale,
                return_embed = True
            )

            self_cond_embed = embed if self.self_cond else None

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            is_mask = ids == self.mask_id

            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    text_embeds = text_embeds,
                    conditioning_token_ids = cond_ids,
                    cond_scale = cond_scale
                )

                scores = rearrange(scores, '... 1 -> ...')

                scores = scores + (uniform(scores.shape, device = device) - 0.5) * critic_noise_scale * (steps_until_x0 / timesteps)

            else:
                probs_without_temperature = logits.softmax(dim = -1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, '... 1 -> ...')

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert self.no_mask_token_prob > 0., 'without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token'

        # get ids

        ids = rearrange(ids, 'b (i j) -> b i j', i = fmap_size, j = fmap_size)

        if not exists(self.vae):
            return ids

        images = self.vae.decode_from_ids(ids)
        return images

    def forward(
        self,
        music_ids: torch.Tensor,
        ignore_index = -1,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
        train_only_generator = False,
        sample_temperature = None
    ):


        # get some basic variables
        batch, n_q, seq_len, device, cond_drop_prob = *music_ids.shape, music_ids.device, default(cond_drop_prob, self.cond_drop_prob)

        index = random.randrange(n_q)
        # prepare mask
        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device = device).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        
        mask_id = self.transformer.mask_id
        labels = torch.where(mask, music_ids[:, index, :], ignore_index)

        if self.no_mask_token_prob > 0.:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, mask_id, music_ids[:, index, :]) 
        raw_music_ids = music_ids[:, index, :].clone().detach()
        music_ids[:, index, :] = x

        # get text embeddings

        if exists(texts):
            text_embeds = self.transformer.encode_text(texts)
            texts = None

        # self conditioning

        self_cond_embed = None

        if self.transformer.self_cond and random.random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    text_embeds = text_embeds,
                    cond_drop_prob = 0.,
                    return_embed = True
                )

                self_cond_embed.detach_()

        # get loss

        ce_loss, logits = self.transformer(
            music_ids,
            index,
            text_embeds = text_embeds,
            self_cond_embed = self_cond_embed,
            labels = labels,
            cond_drop_prob = cond_drop_prob,
            ignore_index = ignore_index,
            return_logits = True
        )

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss
        sample_temperature = sample_temperature if sample_temperature is not None else random.random()
        sampled_ids = gumbel_sample(logits, temperature = sample_temperature)

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (raw_music_ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            text_embeds = text_embeds,
            labels = critic_labels,
            cond_drop_prob = cond_drop_prob
        )

        return ce_loss + self.critic_loss_weight * bce_loss



