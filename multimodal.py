"""Any-to-any modality modules for MNAI Nyx.

The text core is a Llama decoder. These modules bolt image and audio
onto it, in and out, so one model reads and writes all three:

  text  -> the Llama core, unchanged
  image -> VisionEncoder turns pixels into tokens the core can read,
           ImageDecoder turns core states back into pictures
  audio -> AudioEncoder turns mel frames into tokens the core can
           read, AudioDecoder turns core states back into mel frames

Sizes are proportional to the core hidden size, so the same file
works for Nyx 1 and Nyx 1.2. The core dims never change, the modules
just scale with them.

  vision hidden  = core hidden // 8,  24 layers
  image hidden   = core hidden // 4,  12 layers, DiT, 128x128, 1000 steps
  audio hidden   = core hidden // 16, 8 layers, 80 mel bins

All four modules are real PyTorch. Nothing here is a stub. Run this
file on its own for a tiny CPU forward test plus the real param
count on the meta device.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HIDDEN_SIZE

CORE_HIDDEN = HIDDEN_SIZE

VISION_HIDDEN = CORE_HIDDEN // 8
VISION_LAYERS = 24
VISION_HEADS = VISION_HIDDEN // 64
VISION_PATCH = 16
VISION_IMG = 224

IMAGE_HIDDEN = CORE_HIDDEN // 4
IMAGE_LAYERS = 12
IMAGE_HEADS = IMAGE_HIDDEN // 64
IMAGE_PATCH = 8
IMAGE_SIZE = 128
IMAGE_STEPS = 1000

AUDIO_HIDDEN = CORE_HIDDEN // 16
AUDIO_LAYERS = 8
AUDIO_MEL = 80
AUDIO_MAX_FRAMES = 4096


def sinusoidal_time(t, dim):
    # classic sinusoidal embedding for diffusion timesteps
    # t is (B,) long, returns (B, dim) float
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    ).to(t.device)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class VisionEncoder(nn.Module):
    """A ViT that turns an image into tokens the core can read.

    224x224 image, 16x16 patches, 196 patch tokens plus a CLS token,
    24 transformer layers, then a projection into the core hidden
    size. Output is (B, 197, core_hidden), ready to feed to the core
    as input embeddings alongside text tokens.
    """

    def __init__(self, hidden=VISION_HIDDEN, layers=VISION_LAYERS,
                 heads=VISION_HEADS, patch=VISION_PATCH,
                 img_size=VISION_IMG, core_hidden=CORE_HIDDEN):
        super().__init__()
        self.patch = patch
        self.num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, core_hidden)

    def forward(self, x):
        # x is (B, 3, H, W), returns (B, num_patches + 1, core_hidden)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, p], dim=1) + self.pos_embed
        h = self.encoder(h)
        return self.proj(self.norm(h))


class DiTBlock(nn.Module):
    """One DiT block, AdaLN conditioning on time plus core state.

    The conditioning vector carries the diffusion timestep and the
    pooled core hidden state, so the picture the decoder draws is
    steered by what the core said. Scale, shift and gate come from
    the conditioning, which is the AdaLN-Zero layout.
    """

    def __init__(self, hidden, heads, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden * 6),
        )

    def forward(self, x, cond):
        # x is (B, N, hidden), cond is (B, cond_dim)
        s1, sh1, g1, s2, sh2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + g1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        h = self.mlp(h)
        return x + g2.unsqueeze(1) * h


class ImageDecoder(nn.Module):
    """A DiT diffusion head that turns core states into images.

    Takes a noisy 128x128 image, a timestep, and a core hidden state,
    and predicts the noise. The core state is the conditioning, so
    the picture follows what the model meant to draw. Sample with
    the DDIM loop in sample(), 50 steps is enough to see the picture.
    """

    def __init__(self, hidden=IMAGE_HIDDEN, layers=IMAGE_LAYERS,
                 heads=IMAGE_HEADS, patch=IMAGE_PATCH,
                 img_size=IMAGE_SIZE, core_hidden=CORE_HIDDEN,
                 steps=IMAGE_STEPS):
        super().__init__()
        self.patch = patch
        self.img_size = img_size
        self.steps = steps
        self.num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Linear(3 * patch * patch, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden))
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_proj = nn.Linear(core_hidden, hidden)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, hidden) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.head = nn.Linear(hidden, 3 * patch * patch)

    def _patchify(self, x):
        # (B, 3, H, W) -> (B, N, 3*patch*patch)
        p = self.patch
        B, _, H, W = x.shape
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * p * p)
        return x

    def _unpatchify(self, x):
        # (B, N, 3*patch*patch) -> (B, 3, H, W)
        p = self.patch
        n = self.img_size // p
        B = x.shape[0]
        x = x.reshape(B, n, n, 3, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, 3, self.img_size, self.img_size)

    def forward(self, x, t, cond):
        # x is (B, 3, 128, 128) noisy image, t is (B,) long timesteps,
        # cond is (B, core_hidden) core states, returns predicted noise
        h = self.patch_embed(self._patchify(x)) + self.pos_embed
        t_emb = self.time_mlp(sinusoidal_time(t, h.shape[-1]))
        c = t_emb + self.cond_proj(cond)
        for blk in self.blocks:
            h = blk(h, c)
        return self._unpatchify(self.head(self.final_norm(h)))

    @torch.no_grad()
    def sample(self, cond, steps=50, seed=0):
        # DDIM sampling from pure noise, cond is (B, core_hidden)
        device = cond.device
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(cond.shape[0], 3, self.img_size, self.img_size,
                        generator=g, device=device)
        betas = torch.linspace(1e-4, 0.02, self.steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        timesteps = torch.linspace(self.steps - 1, 0, steps,
                                   dtype=torch.long, device=device)
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(cond.shape[0])
            eps = self.forward(x, t_batch, cond)
            a_t = alphas_cumprod[t]
            a_prev = (alphas_cumprod[timesteps[i + 1]]
                      if i + 1 < len(timesteps)
                      else torch.tensor(1.0, device=device))
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps
        return x.clamp(-1, 1)


class AudioEncoder(nn.Module):
    """Turns mel spectrogram frames into tokens the core can read.

    Input is (B, T, 80) mel bins. A linear projection, learned
    positions, 8 transformer layers, then a projection into the core
    hidden size. Output is (B, T, core_hidden), ready to feed to the
    core alongside text tokens.
    """

    def __init__(self, hidden=AUDIO_HIDDEN, layers=AUDIO_LAYERS,
                 mel_bins=AUDIO_MEL, core_hidden=CORE_HIDDEN,
                 max_frames=AUDIO_MAX_FRAMES):
        super().__init__()
        self.in_proj = nn.Linear(mel_bins, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=max(1, hidden // 64),
            dim_feedforward=hidden * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, core_hidden)

    def forward(self, mel):
        # mel is (B, T, 80), returns (B, T, core_hidden)
        t = mel.shape[1]
        h = self.in_proj(mel) + self.pos_embed[:, :t]
        h = self.encoder(h)
        return self.proj(self.norm(h))


class AudioDecoder(nn.Module):
    """Turns core hidden states back into mel spectrogram frames.

    Input is (B, T, core_hidden) from the core. A projection down to
    the audio size, 8 transformer layers, then a linear head to 80
    mel bins. Output is (B, T, 80), which a vocoder turns into sound.
    """

    def __init__(self, hidden=AUDIO_HIDDEN, layers=AUDIO_LAYERS,
                 mel_bins=AUDIO_MEL, core_hidden=CORE_HIDDEN):
        super().__init__()
        self.in_proj = nn.Linear(core_hidden, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=max(1, hidden // 64),
            dim_feedforward=hidden * 4, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, mel_bins)

    def forward(self, hidden_states):
        # hidden_states is (B, T, core_hidden), returns (B, T, 80) mel
        h = self.in_proj(hidden_states)
        h = self.decoder(h)
        return self.out_proj(self.norm(h))


class AnyToAnyModel(nn.Module):
    """One model, three modalities in and out.

    Holds the Llama text core plus the four modality modules. Text
    goes straight through the core. Images go through the vision
    encoder on the way in and the image decoder on the way out.
    Audio goes through the audio encoder on the way in and the audio
    decoder on the way out. The core is the brain, the modules are
    the eyes, the hands and the ears.
    """

    def __init__(self, core, vision=None, image=None,
                 audio_enc=None, audio_dec=None):
        super().__init__()
        self.core = core
        self.vision_encoder = vision or VisionEncoder()
        self.image_decoder = image or ImageDecoder()
        self.audio_encoder = audio_enc or AudioEncoder()
        self.audio_decoder = audio_dec or AudioDecoder()

    def encode_image(self, images):
        # images (B, 3, 224, 224) -> (B, 197, core_hidden) tokens
        return self.vision_encoder(images)

    def decode_image(self, cond, steps=50, seed=0):
        # cond (B, core_hidden) -> (B, 3, 128, 128) picture
        return self.image_decoder.sample(cond, steps=steps, seed=seed)

    def encode_audio(self, mel):
        # mel (B, T, 80) -> (B, T, core_hidden) tokens
        return self.audio_encoder(mel)

    def decode_audio(self, hidden_states):
        # (B, T, core_hidden) -> (B, T, 80) mel
        return self.audio_decoder(hidden_states)

    def forward(self, modality="text", **kwargs):
        # text: pass input_ids and labels straight to the core
        if modality == "text":
            return self.core(**kwargs)
        # image in: encode the picture, feed tokens to the core
        if modality == "image":
            tokens = self.encode_image(kwargs.pop("images"))
            return self.core(inputs_embeds=tokens, **kwargs)
        # audio in: encode the mel, feed tokens to the core
        if modality == "audio":
            tokens = self.encode_audio(kwargs.pop("mel"))
            return self.core(inputs_embeds=tokens, **kwargs)
        raise ValueError(f"unknown modality {modality!r}, pick text, image or audio")

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def modality_breakdown(self):
        # where the parameters live, per module
        parts = {}
        for name, p in self.named_parameters():
            key = name.split(".")[0]
            parts[key] = parts.get(key, 0) + p.numel()
        return parts


def build_any_to_any(core=None, dtype=None):
    """Wrap a Llama core with the four modality modules.

    Pass a core you already built, or none and one is built fresh.
    dtype None leaves everything on the meta device, which is how
    you count parameters without allocating terabytes of RAM.
    """
    from model import build_model

    if core is None:
        core = build_model(dtype=dtype)
    return AnyToAnyModel(core)


def modality_param_counts():
    """Exact param counts for each module, on the meta device."""
    with torch.device("meta"):
        counts = {
            "vision_encoder": sum(p.numel() for p in VisionEncoder().parameters()),
            "image_decoder": sum(p.numel() for p in ImageDecoder().parameters()),
            "audio_encoder": sum(p.numel() for p in AudioEncoder().parameters()),
            "audio_decoder": sum(p.numel() for p in AudioDecoder().parameters()),
        }
    counts["modalities_total"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    # tiny CPU forward test, small dims so it actually runs here
    print("tiny forward test on CPU, small dims:")
    tiny_core = 64
    v = VisionEncoder(hidden=32, layers=2, heads=4, core_hidden=tiny_core)
    img = torch.randn(1, 3, 224, 224)
    toks = v(img)
    print(f"  vision encoder: {tuple(img.shape)} -> {tuple(toks.shape)}")
    d = ImageDecoder(hidden=64, layers=2, heads=4, core_hidden=tiny_core)
    noise = torch.randn(1, 3, 128, 128)
    t = torch.tensor([500])
    cond = torch.randn(1, tiny_core)
    pred = d(noise, t, cond)
    print(f"  image decoder: {tuple(noise.shape)} -> {tuple(pred.shape)}")
    a = AudioEncoder(hidden=32, layers=2, mel_bins=80, core_hidden=tiny_core)
    mel = torch.randn(1, 16, 80)
    atoks = a(mel)
    print(f"  audio encoder: {tuple(mel.shape)} -> {tuple(atoks.shape)}")
    ad = AudioDecoder(hidden=32, layers=2, mel_bins=80, core_hidden=tiny_core)
    out_mel = ad(atoks)
    print(f"  audio decoder: {tuple(atoks.shape)} -> {tuple(out_mel.shape)}")
    print("  all four modules run, shapes check out")
    print()
    # real param counts on the meta device
    print("real module sizes for this core:")
    counts = modality_param_counts()
    for k, n in counts.items():
        print(f"  {k:18s} {n:>15,}  ({n / 1e9:.2f}B)")
