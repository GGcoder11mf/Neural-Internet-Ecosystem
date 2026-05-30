"""
Computation backend for the Fake Internet Ecosystem.

PyTorch CUDA implementation with neural network augmentation.
Falls back to NumPy if PyTorch is not available.
Uses GPU tensors for vectorized operations when CUDA is available.
"""

import numpy as np
import os

# Try importing torch for GPU acceleration
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = None


def _to_tensor(arr):
    """Convert numpy array to torch tensor on GPU if available."""
    if TORCH_AVAILABLE:
        return torch.tensor(np.asarray(arr, dtype=np.float32), device=DEVICE)
    return np.asarray(arr, dtype=np.float64)


def _from_tensor(t):
    """Convert torch tensor back to numpy array."""
    if TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)


def compute_attention_matrix(view_counts, engagement_rates, novelty_scores, tick, novelty_decay=0.95):
    """
    Compute attention scores for all posts using GPU-accelerated operations.
    
    attention = views * engagement_rate * novelty_decay^(tick - created_tick)
    """
    if TORCH_AVAILABLE:
        views = torch.tensor(np.asarray(view_counts, dtype=np.float32), device=DEVICE)
        engagement = torch.tensor(np.asarray(engagement_rates, dtype=np.float32), device=DEVICE)
        novelty = torch.tensor(np.asarray(novelty_scores, dtype=np.float32), device=DEVICE)
        
        views_factor = torch.log1p(views)
        attention = views_factor * (1 + engagement) * novelty
        attention = torch.clamp(attention, min=0.001)
        return attention.cpu().numpy()
    else:
        # NumPy fallback
        views_factor = np.log1p(np.asarray(view_counts, dtype=np.float64))
        engagement = np.asarray(engagement_rates, dtype=np.float64)
        novelty = np.asarray(novelty_scores, dtype=np.float64)
        attention = views_factor * (1 + engagement) * novelty
        return np.maximum(attention, 0.001)


def compute_similarity_matrix(vectors_a, vectors_b=None):
    """
    Compute cosine similarity matrix between two sets of vectors.
    Uses GPU tensors if available.
    """
    if TORCH_AVAILABLE:
        a = torch.tensor(np.asarray(vectors_a, dtype=np.float32), device=DEVICE)
        if vectors_b is None:
            b = a
        else:
            b = torch.tensor(np.asarray(vectors_b, dtype=np.float32), device=DEVICE)
        
        norms_a = torch.norm(a, dim=1, keepdim=True).clamp(min=1e-8)
        norms_b = torch.norm(b, dim=1, keepdim=True).clamp(min=1e-8)
        
        normalized_a = a / norms_a
        normalized_b = b / norms_b
        
        result = normalized_a @ normalized_b.T
        return result.cpu().numpy()
    else:
        vectors_a = np.asarray(vectors_a, dtype=np.float64)
        if vectors_b is None:
            vectors_b = vectors_a
        else:
            vectors_b = np.asarray(vectors_b, dtype=np.float64)
        
        norms_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
        norms_a = np.maximum(norms_a, 1e-8)
        norms_b = np.maximum(norms_b, 1e-8)
        
        normalized_a = vectors_a / norms_a
        normalized_b = vectors_b / norms_b
        
        return normalized_a @ normalized_b.T


def batch_cosine_similarity(vec_a, matrix_b):
    """
    Compute cosine similarity between one vector and a matrix of vectors.
    """
    if TORCH_AVAILABLE:
        va = torch.tensor(np.asarray(vec_a, dtype=np.float32), device=DEVICE)
        mb = torch.tensor(np.asarray(matrix_b, dtype=np.float32), device=DEVICE)
        
        norm_a = torch.norm(va).clamp(min=1e-8)
        norms_b = torch.norm(mb, dim=1).clamp(min=1e-8)
        
        dots = mb @ va
        result = dots / (norm_a * norms_b)
        return result.cpu().numpy()
    else:
        vec_a = np.asarray(vec_a, dtype=np.float64)
        matrix_b = np.asarray(matrix_b, dtype=np.float64)
        
        norm_a = max(np.linalg.norm(vec_a), 1e-8)
        norms_b = np.linalg.norm(matrix_b, axis=1)
        norms_b = np.maximum(norms_b, 1e-8)
        
        dots = matrix_b @ vec_a
        return dots / (norm_a * norms_b)


def rank_feed(attention_scores, agent_interests, post_topics, platform_bias, 
              network_distances, echo_chamber_strength=0.5, 
              virality_sensitivity=1.0, nn_predictions=None):
    """
    Rank posts for a specific agent's feed.
    
    feed_score = attention * virality_sensitivity + 
                 interest_similarity * echo_chamber_strength +
                 platform_bias * proximity +
                 nn_prediction * nn_weight
    """
    if TORCH_AVAILABLE:
        att = torch.tensor(np.asarray(attention_scores, dtype=np.float32), device=DEVICE)
        agent_int = torch.tensor(np.asarray(agent_interests, dtype=np.float32), device=DEVICE)
        p_topics = torch.tensor(np.asarray(post_topics, dtype=np.float32), device=DEVICE)
        p_bias = torch.tensor(np.asarray(platform_bias, dtype=np.float32), device=DEVICE)
        net_dist = torch.tensor(np.asarray(network_distances, dtype=np.float32), device=DEVICE)
        
        # Interest similarity
        norm_agent = torch.norm(agent_int).clamp(min=1e-8)
        norms_topics = torch.norm(p_topics, dim=1).clamp(min=1e-8)
        interest_sim = (p_topics @ agent_int) / (norm_agent * norms_topics)
        
        # Network proximity
        max_dist = torch.max(net_dist).clamp(min=1e-8)
        proximity = 1.0 - (net_dist / max_dist)
        
        # Combined feed score
        feed_score = (att * virality_sensitivity + 
                     interest_sim * echo_chamber_strength +
                     p_bias * proximity)
        
        # Add neural network predictions if available
        if nn_predictions is not None:
            nn_pred = torch.tensor(np.asarray(nn_predictions, dtype=np.float32), device=DEVICE)
            feed_score += nn_pred * 0.3  # NN weight
        
        ranked_indices = torch.argsort(-feed_score)
        return ranked_indices.cpu().numpy().astype(int), feed_score.cpu().numpy()
    else:
        attention_scores = np.asarray(attention_scores, dtype=np.float64)
        agent_interests = np.asarray(agent_interests, dtype=np.float64)
        post_topics = np.asarray(post_topics, dtype=np.float64)
        platform_bias = np.asarray(platform_bias, dtype=np.float64)
        network_distances = np.asarray(network_distances, dtype=np.float64)
        
        norm_agent = max(np.linalg.norm(agent_interests), 1e-8)
        norms_topics = np.linalg.norm(post_topics, axis=1)
        norms_topics = np.maximum(norms_topics, 1e-8)
        interest_sim = (post_topics @ agent_interests) / (norm_agent * norms_topics)
        
        max_dist = max(np.max(network_distances), 1e-8)
        proximity = 1.0 - (network_distances / max_dist)
        
        feed_score = (attention_scores * virality_sensitivity + 
                     interest_sim * echo_chamber_strength +
                     platform_bias * proximity)
        
        if nn_predictions is not None:
            feed_score += np.asarray(nn_predictions, dtype=np.float64) * 0.3
        
        ranked_indices = np.argsort(-feed_score)
        return ranked_indices.astype(int), feed_score


def compute_community_affinity(agent_vectors, n_clusters=None):
    """
    Compute community affinity matrix using GPU-accelerated operations.
    Returns a similarity matrix that can be used for clustering.
    """
    return compute_similarity_matrix(agent_vectors)
