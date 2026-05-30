"""
GPU computation backend for the Fake Internet Ecosystem.

Uses PyTorch for CUDA acceleration when available.
Falls back to NumPy when PyTorch is not installed.
"""

from .backend import (
    compute_attention_matrix,
    compute_similarity_matrix,
    batch_cosine_similarity,
    rank_feed,
    compute_community_affinity,
)

from .neural_net import (
    FeedRankingNet,
    NeuralRankingEngine,
    TORCH_AVAILABLE,
    CUDA_AVAILABLE,
    DEVICE,
)

__all__ = [
    "compute_attention_matrix",
    "compute_similarity_matrix",
    "batch_cosine_similarity",
    "rank_feed",
    "compute_community_affinity",
    "FeedRankingNet",
    "NeuralRankingEngine",
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE",
    "DEVICE",
]
