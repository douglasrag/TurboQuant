"""
Lloyd-Max codebook builder in MLX, d = 64 (Qwen3.5 0.8B head dim)

Run once to build the codebook, then the frozen constants are used in the model
for quantization. 
"""

import mlx.core as mx
import math
import numpy as np

# polar transformation 

def polar_transformation(x: np.ndarray): 
    angles = []
    vec = x.copy()

    while len(vec) > 1:
        x1, x2 = vec[0], vec[1]
        angles.append(np.arctan2(x2, x1))
        vec = np.concatenate([[np.sqrt(x1**2 + x2**2)], vec[2:]])

    return np.array(angles), vec[0]

# batched angle sampling 
def sample_angles(dim: int = 64, n_samples: int = 200_000) -> np.ndarray: 
    print(f"Sampling {n_samples} angles for dim {dim}")
    all_angles = []
    chunk = 5_000

    for start in range(0, n_samples, chunk): 
        n = min(chunk, n_samples - start)
        
        vecs_mlx = mx.random.normal(shape = (n, dim))
        mx.eval(vecs_mlx)

        vecs_np = np.array(vecs_mlx.tolist())

        for i in range(n):
            a, _ = polar_transformation(vecs_np[i])
            all_angles.append(a)

        if start % 50_000 == 0:
            print(f" {start:,} / {n_samples:,} angles sampled")

    samples = np.concatenate(all_angles)
    print(f" Total angles sampled: {samples.shape[0]:,}")
    return samples

# Lloyd-Max iteration
def build_lloyd_max_codebook_mlx(
    n_bins: int = 8,
    dim: int = 64,
    n_samples: int = 200_000,
) -> tuple[mx.array, mx.array]:

    samples = sample_angles(dim, n_samples)
    print(f" Total angles sampled: {samples.shape[0]:,}")

    pi = math.pi
    centroids = mx.linspace(-pi, pi, n_bins)

    for iteration in range(500): 
        boundaries = np.concatenate([
            [-pi], (centroids[:-1] + centroids[1:]) / 2, [pi]
        ])
        new_centroids = np.array([
            samples[(samples >= boundaries[i]) & (samples < boundaries[i+1])].mean()
            if ((samples >= boundaries[i]) & (samples < boundaries[i + 1])).sum() > 0
            else centroids[i]
            for i in range(n_bins)
        ])
        delta = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if delta < 1e-8: 
            print(f" Converged in {iteration} iterations")
            break

    boundaries = np.concatenate([
        [-pi], (centroids[:-1] + centroids[1:]) / 2, [pi]
    ])
    return mx.array(centroids.tolist()), mx.array(boundaries.tolist())

def quantize_angles(angles: mx.array, boundaries: mx.array) -> mx.array:
    return mx.searchsorted(boundaries[1:-1], angles)

def dequantize_angles(indicies: mx.array, centroids: mx.array) -> mx.array:
    return centroids[indicies]

# run loop (main)

if __name__ == "__main__":
    print("Building Lloyd-Max codebook...")\
    
    centroids, boundaries = build_lloyd_max_codebook_mlx(
        n_bins = 8,
        dim = 64,
        n_samples = 200_000,
    )

    c = [round(x, 6) for x in centroids.tolist()]
    b = [round(x, 6) for x in boundaries.tolist()]
    print(f"Codebook centroids: {c}")
    print(f"Codebook boundaries: {b}")

    print("Paste these frozen constants into your pipeline:\n")
    print(f"CODEBOOK  = mx.array({c})")
    print(f"BOUNDARIES = mx.array({b})")
    print("\nCodebook summary:")
    print(f"  centroids:  {[round(x, 4) for x in c]}")
    print(f"  boundaries: {[round(x, 4) for x in b]}")
 