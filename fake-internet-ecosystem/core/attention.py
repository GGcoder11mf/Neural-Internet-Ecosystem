"""
Attention mechanics engine for the Fake Internet Ecosystem.

Attention is the core resource that flows through content like electricity.
This module implements the attention scoring, decay, distribution system,
controversy detection, and engagement bait scoring.

Realism improvements:
- Controversy boost (platforms amplify controversial content)
- Ratio detection (replies >> likes = controversy signal)
- Engagement bait scoring (content designed to provoke reactions)
- Better virality cascade modeling (SIR-like spread)
- Doom-scroll amplification (negative content traps attention)
- Attention inequality tracking (Gini coefficient)
"""

import numpy as np
from typing import List, Dict, Tuple
from .content import Post
from .agent import UserAgent


class AttentionEngine:
    """
    The attention engine computes how attention flows through the ecosystem.
    
    Key formulas:
        attention(post) = views * engagement_rate * novelty_decay^age * controversy_boost
        visibility(post, agent) = f(attention, platform_bias, network_distance, doom_factor)
        
    Attention is a limited resource flowing through content like electricity.
    Controversial content gets a platform boost because engagement = revenue.
    """
    
    def __init__(self, novelty_decay: float = 0.95, 
                 attention_decay: float = 0.9,
                 min_attention: float = 0.001,
                 virality_threshold: float = 5.0,
                 controversy_amplification: float = 1.3,
                 engagement_bait_weight: float = 0.15):
        self.novelty_decay = novelty_decay
        self.attention_decay = attention_decay
        self.min_attention = min_attention
        self.virality_threshold = virality_threshold
        self.controversy_amplification = controversy_amplification
        self.engagement_bait_weight = engagement_bait_weight
        
        # Attention history for tracking
        self.total_attention_history: List[float] = []
        self.inequality_history: List[float] = []
    
    def compute_post_attention(self, post: Post, current_tick: int) -> float:
        """
        Compute attention score for a single post.
        
        attention = views * engagement_rate * novelty_decay^age * controversy_boost
        """
        age = post.age_at(current_tick)
        novelty = self.novelty_decay ** age
        
        engagement_rate = post.engagement_rate
        views_factor = np.log1p(post.views)
        
        # Base attention
        attention = views_factor * (1 + engagement_rate) * novelty
        
        # Controversy boost: controversial content gets amplified
        # This models how platforms boost content that drives engagement
        controversy = post.controversy_score
        if controversy > 0.3:
            controversy_boost = 1 + (controversy - 0.3) * self.controversy_amplification
            attention *= controversy_boost
        
        # Engagement bait boost: high reply-to-like ratio signals provocative content
        if post.ratio_score > 2.0:
            bait_boost = 1 + self.engagement_bait_weight * min(post.ratio_score, 10)
            attention *= bait_boost
        
        # Apply decay to previous attention
        post.attention_score = max(attention, self.min_attention)
        post.record_attention(post.attention_score)
        
        return post.attention_score
    
    def compute_all_attention(self, posts: List[Post], current_tick: int) -> np.ndarray:
        """Compute attention scores for all posts using vectorized operations."""
        if not posts:
            return np.array([])
        
        n = len(posts)
        views = np.zeros(n)
        engagement_rates = np.zeros(n)
        novelty_scores = np.zeros(n)
        controversy_scores = np.zeros(n)
        ratio_scores = np.zeros(n)
        
        for i, post in enumerate(posts):
            views[i] = post.views
            engagement_rates[i] = post.engagement_rate
            age = post.age_at(current_tick)
            novelty_scores[i] = self.novelty_decay ** age
            controversy_scores[i] = post.controversy_score
            ratio_scores[i] = post.ratio_score
        
        # Vectorized computation
        views_factor = np.log1p(views)
        attention = views_factor * (1 + engagement_rates) * novelty_scores
        
        # Controversy boost (vectorized)
        controversy_mask = controversy_scores > 0.3
        controversy_boost = np.ones(n)
        controversy_boost[controversy_mask] = 1 + (controversy_scores[controversy_mask] - 0.3) * self.controversy_amplification
        attention *= controversy_boost
        
        # Engagement bait boost (vectorized)
        bait_mask = ratio_scores > 2.0
        bait_boost = np.ones(n)
        bait_boost[bait_mask] = 1 + self.engagement_bait_weight * np.minimum(ratio_scores[bait_mask], 10)
        attention *= bait_boost
        
        attention = np.maximum(attention, self.min_attention)
        
        # Update posts
        for i, post in enumerate(posts):
            post.attention_score = float(attention[i])
            post.record_attention(float(attention[i]))
            # Update controversy
            post.compute_controversy()
        
        return attention
    
    def compute_visibility(self, post_attention: float, platform_bias: float,
                          network_distance: float, echo_chamber: float = 0.5,
                          doom_factor: float = 0.0) -> float:
        """
        Compute visibility of a post for a specific agent.
        
        visibility = attention * (1 + platform_bias * (1 - network_distance) * echo_chamber)
                     * (1 + doom_factor)  # doom scrollers see more negative content
        """
        distance_factor = 1.0 / (1.0 + network_distance)
        visibility = post_attention * (1.0 + platform_bias * distance_factor * echo_chamber)
        
        # Doom scroll amplification: if the agent is in a doom scroll, they see
        # more negative/controversial content (algorithm feeds the spiral)
        if doom_factor > 0:
            visibility *= (1.0 + doom_factor * 0.3)
        
        return visibility
    
    def compute_network_distance(self, agent: UserAgent, post_author_id: str,
                                  agent_map: Dict[str, UserAgent]) -> float:
        """
        Compute social network distance between an agent and a post's author.
        
        Distance = 0 if following, 1 if 2nd degree, 2+ based on interest divergence.
        """
        if post_author_id == agent.id:
            return 0.0
        
        if post_author_id in agent.following:
            return 0.2  # Direct connection
        
        # Check 2nd degree
        if post_author_id in agent_map:
            author = agent_map[post_author_id]
            for mutual_id in agent.following:
                if mutual_id in agent_map and post_author_id in agent_map[mutual_id].following:
                    return 0.5  # 2nd degree
        
        # Use interest divergence as proxy
        if post_author_id in agent_map:
            author = agent_map[post_author_id]
            similarity = np.dot(agent.interests, author.interests)
            return max(1.0 - similarity, 0.3)
        
        return 1.0  # Unknown / distant
    
    def distribute_attention(self, agents: List[UserAgent], posts: List[Post],
                             attention_scores: np.ndarray, platform_biases: np.ndarray,
                             agent_map: Dict[str, UserAgent],
                             echo_chamber_strength: float = 0.5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Distribute attention across agents based on feed ranking.
        
        Returns: agent_id -> [(post_id, visibility_score), ...] ranked feed
        """
        feeds: Dict[str, List[Tuple[str, float]]] = {}
        
        for agent in agents:
            feed_items = []
            doom_factor = min(agent.doom_scroll_depth, 5) / 5.0 if hasattr(agent, 'doom_scroll_depth') else 0.0
            
            for i, post in enumerate(posts):
                network_dist = self.compute_network_distance(agent, post.author_id, agent_map)
                visibility = self.compute_visibility(
                    attention_scores[i], platform_biases[i],
                    network_dist, echo_chamber_strength, doom_factor
                )
                feed_items.append((post.id, visibility, i))
            
            # Sort by visibility (descending)
            feed_items.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to attention budget
            budget = agent.params["attention_budget"]
            feed_items = feed_items[:budget]
            
            feeds[agent.id] = [(item[0], item[1]) for item in feed_items]
        
        return feeds
    
    def compute_attention_inequality(self, attention_scores: np.ndarray) -> float:
        """
        Compute Gini coefficient of attention distribution.
        Like wealth inequality but for views.
        """
        if len(attention_scores) == 0:
            return 0.0
        
        scores = np.sort(attention_scores)
        n = len(scores)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * scores) - (n + 1) * np.sum(scores)) / (n * np.sum(scores) + 1e-10)
        return np.clip(gini, 0, 1)
    
    def is_viral(self, attention_score: float) -> bool:
        """Check if an attention score exceeds virality threshold."""
        return attention_score > self.virality_threshold
    
    def compute_virality_cascade_probability(self, post: Post, n_exposed: int,
                                              engagement_rate: float) -> float:
        """
        Model virality as an SIR-like cascade.
        
        Each exposure has a probability of "infecting" a new person
        (getting them to share it). The probability depends on:
        - Engagement rate (higher = more likely to share)
        - Already-exposed count (saturation effect)
        - Content novelty (fresh content spreads faster)
        
        This models how real viral content spreads: each person who sees it
        has a small chance of amplifying it further, creating a cascade.
        """
        if n_exposed < 1:
            return 0.0
        
        # Base cascade probability from engagement
        base_prob = engagement_rate * 0.1
        
        # Novelty bonus (fresh content spreads faster)
        novelty_bonus = post.novelty_score * 0.05
        
        # Saturation effect: as more people see it, each new exposure is less impactful
        saturation_factor = 1.0 / (1.0 + np.log1p(n_exposed))
        
        cascade_prob = (base_prob + novelty_bonus) * saturation_factor
        
        # Controversy increases cascade probability (people can't help but share)
        if post.controversy_score > 0.3:
            cascade_prob *= 1.0 + post.controversy_score * 0.5
        
        return min(cascade_prob, 0.3)  # Cap at 30% per exposure
    
    def compute_virality_curve(self, post: Post) -> List[float]:
        """Get the virality curve (attention over time) for a post."""
        return post.attention_history
