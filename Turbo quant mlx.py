import mlx.core as mx
from mlx_lm import load
import numpy as np
import math

# Frozen CODEBOOK, precomputed from lloyd-max and is used throughout the model for quantization. 
CODEBOOK   = mx.array([-2.360208, -0.803322, -0.307282, -0.08846,
                         0.087261,  0.305318,  0.799506,  2.357275])
BOUNDARIES = mx.array([-3.141593, -1.581765, -0.555302, -0.197871,
                        -0.0006,    0.19629,   0.552412,  1.57839,
                         3.141593])

HEAD_DIM   = 64
N_KV_HEADS = 8

# Polar Transformation: converts the vectors into radius and angle pairs. 

def polar_transformation(x: np.ndarray):
    angles = []
    vec = x.copy()
    while len(vec) > 1:
        x1, x2 = vec[0], vec[1]
        angles.append(np.arctan2(x2, x1))
        vec = np.concatenate([[np.sqrt(x1**2 + x2**2)], vec[2:]])
    return np.array(angles), vec[0]

def polar_to_cartesian(angles: np.ndarray, norm: float) -> np.ndarray:
    vec = np.array([norm])
    for theta in reversed(angles):
        r = vec[0]
        vec = np.concatenate([[r * np.cos(theta), r * np.sin(theta)], vec[1:]])
    return vec

# quantize & dequantize 

def quantize_angles(angles: np.ndarray) -> np.ndarray:
    bnds = np.array(BOUNDARIES.tolist())
    return np.searchsorted(bnds[1:-1], angles)

def dequantize_angles(indices: np.ndarray) -> np.ndarray:
    cents = np.array(CODEBOOK.tolist())
    return cents[indices]

# rotation 

def make_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    R, _ = np.linalg.qr(A)
    return R

# seperating read & write operation for more understanding the logic sake. 

def turbo_write(k_vec: np.ndarray, R: np.ndarray) -> dict:
    k_rot        = R @ k_vec
    angles, norm = polar_transformation(k_rot)
    indices      = quantize_angles(angles)
    angles_q     = dequantize_angles(indices)
    k_recon      = polar_to_cartesian(angles_q, norm)
    signs        = np.sign(k_rot - k_recon)
    return {"indices": indices, "norm": norm, "signs": signs}

def turbo_read(q_vec: np.ndarray, cache: dict, R: np.ndarray) -> float:
    angles_q   = dequantize_angles(cache["indices"])
    k_recon    = polar_to_cartesian(angles_q, cache["norm"])
    Q_rot      = R @ q_vec
    score_main = np.dot(Q_rot, k_recon)
    score_corr = np.dot(Q_rot, cache["signs"])
    scale      = 1.0 / (2 * len(cache["signs"]))
    return score_main + score_corr * scale

# comparison matrics to see before and after changes. 

def bytes_bf16(dim: int, n_heads: int) -> int:
    return dim * n_heads * 2

def bytes_turbo(dim: int, n_heads: int) -> int:
    per_head = math.ceil((dim-1)*3/8) + 4 + math.ceil(dim/8)
    return per_head * n_heads

# main 

def run():
    print("TurboQuant MLX — Qwen3.5-0.8B")
    print("=" * 56)

    print("loading model...")
    model, tokenizer = load("mlx-community/Qwen3.5-0.8B-MLX-bf16")
    print("model loaded.")

    # identify which layer indices have self_attn, different models have different structure, we're targetting self-attn, not lin_attn layers. 
    self_attn_layers = [
        i for i, layer in enumerate(model.layers)
        if hasattr(layer, "self_attn")
    ]
    print(f"self_attn layers: {self_attn_layers}")

    # hook: capture K from self_attn.k_proj output. Why? Qwen doesn't cache the k keys, thus we simply make a hook to capture it and store it to the side.
    # we wrap k_proj.forward so we can intercept the raw K tensor
    # what we're doing is essentially fishing out the raw K tensors before any operation is done on them, for comparing pre- and post- Turbo, which is the whole point. 
    # since this is just a exercize on TurboQuant, I'm not integrating the captured, and quantized value for decision making (aka. predicting tokens), but merely storing them to compare pre- and port- Turbo's effect. 
    captured = {}   # layer_idx -> np.ndarray (n_kv_heads, seq_len, head_dim)

    original_forwards = {}

    def make_hook(layer_idx, attn):
        orig_k_proj = attn.k_proj

        class HookedKProj:
            def __call__(self, x):
                out = orig_k_proj(x)          # (batch, seq_len, n_kv_heads*head_dim)
                mx.eval(out)
                # reshape to (batch, seq_len, n_kv_heads, head_dim)
                b, s, _ = out.shape
                reshaped = out.reshape(b, s, N_KV_HEADS, HEAD_DIM)
                # store as numpy: (n_kv_heads, seq_len, head_dim)
                captured[layer_idx] = np.array(reshaped.tolist()[0]).transpose(1, 0, 2)
                return out

        attn.k_proj = HookedKProj()

    for layer_idx in self_attn_layers:
        make_hook(layer_idx, model.layers[layer_idx].self_attn)

    # forward pass
    prompt  = "The capital of France is"
    tokens  = mx.array(tokenizer.encode(prompt))[None]
    seq_len = tokens.shape[1]
    print(f"prompt : '{prompt}'  |  tokens: {seq_len}")

    from mlx_lm.models.cache import make_prompt_cache
    cache  = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    print(f"forward pass done. captured {len(captured)} K tensors.")

    if not captured:
        print("no K tensors captured — hook didn't fire")
        print("self_attn layer attribute names:")
        for i in self_attn_layers[:2]:
            print(f"  layer {i}:", dir(model.layers[i].self_attn))
        return

    # layer-by-layer comparison
    print("\n--- layer-by-layer comparison ---")
    print(f"{'layer':>6}  {'true score':>12}  {'turbo score':>12}  {'norm err':>10}")
    print("-" * 56)

    results = []

    for layer_idx in sorted(captured.keys()):
        k_np  = captured[layer_idx]   # (n_kv_heads, seq_len, head_dim)
        k_vec = k_np[0, -1, :]        # head 0, last token

        R          = make_rotation_matrix(HEAD_DIM, seed=layer_idx)
        compressed = turbo_write(k_vec, R)

        rng        = np.random.default_rng(layer_idx + 100)
        q_vec      = rng.standard_normal(HEAD_DIM)

        score_true  = np.dot(q_vec, k_vec)
        score_turbo = turbo_read(q_vec, compressed, R)
        denom       = np.linalg.norm(q_vec) * np.linalg.norm(k_vec)
        norm_err    = abs(score_true - score_turbo) / max(denom, 1e-9)

        results.append({"layer": layer_idx, "norm_err": norm_err,
                         "score_true": score_true, "score_turbo": score_turbo})

        print(f"{layer_idx:>6}  {score_true:>12.4f}  {score_turbo:>12.4f}  "
              f"{norm_err:>10.4f}")

    if results:
        errs  = [r["norm_err"] for r in results]
        ratio = bytes_bf16(HEAD_DIM, N_KV_HEADS) / bytes_turbo(HEAD_DIM, N_KV_HEADS)

        print("-" * 56)
        print(f"\nnorm error — mean: {np.mean(errs):.4f}  "
              f"median: {np.median(errs):.4f}  "
              f"max: {np.max(errs):.4f}")
        print(f"\ncompression : {ratio:.2f}x")
        print(f"  BF16      : {bytes_bf16(HEAD_DIM, N_KV_HEADS)} bytes / token")
        print(f"  TurboQuant: {bytes_turbo(HEAD_DIM, N_KV_HEADS)} bytes / token")
        print(f"  self_attn layers: {len(results)}")

if __name__ == "__main__":
    run()