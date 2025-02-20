import numpy as np
from sklearn.random_projection import GaussianRandomProjection

def debug_lsh():
    N = 2048**2  # Total number of vectors
    K = 16       # Dimensionality of each vector
    B = 2048     # Number of buckets (B << N)
    L = 10       # Number of hash functions (typically tuned experimentally)

    # Generate random vectors as input
    np.random.seed(42)  # For reproducibility
    vectors = np.random.randn(N, K)  # Sampled from a normal distribution
    lsh(vectors,N, K,B, L)

def lsh(
    vectors,
    N = 2048**2,  # Total number of vectors
    K = 16,       # Dimensionality of each vector
    B = 2048,     # Number of buckets (B << N)
    L = 10,       # Number of hash functions (typically tuned experimentally)
)

    # Generate L random projection hash functions
    hash_planes = [GaussianRandomProjection(n_components=K) for _ in range(L)]

    # Compute hash values for all vectors at once
    def hash_function_matrix(V):
        """Compute L hash values for all vectors V in a single operation."""
        hashes = np.array([proj.fit_transform(V) > 0 for proj in hash_planes])  # Shape (L, N, K)
        return hashes.astype(int).reshape(L, N, K).sum(axis=2).T  # Summing over K to get L-bit hashes

    # Compute hashes for all vectors
    hashed_values = hash_function_matrix(vectors)  # Shape (N, L)

    # Convert to hashable tuple representation for bucket assignment
    bucket_indices = [tuple(row) for row in hashed_values]

    # Efficient dictionary-based bucketing
    from collections import defaultdict
    hash_table = defaultdict(list)

    # Vectorized bucketing using dictionary mapping
    for i, bucket_index in enumerate(bucket_indices):
        hash_table[bucket_index].append(i)

    # Print some sample bucket sizes
    print(f"Total unique buckets: {len(hash_table)}")
    for k, v in list(hash_table.items())[:5]:
        print(f"Bucket {k}: {len(v)} vectors")
    return hash_table

