"""
Metrics Tracking System for the Fake Internet Ecosystem.

Tracks emergent data:
- Meme lifespan distribution
- Virality curves
- Polarization index
- Language divergence score
- Influencer dominance
- Attention inequality (Gini coefficient for views)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from .agent import UserAgent
from .content import Post, Meme
from .community import CommunityDetector


class MetricsTracker:
    """Tracks and computes all ecosystem metrics over time."""
    
    def __init__(self):
        # Time series data
        self.tick_data: List[Dict] = []
        self.current_tick = 0
        
        # Cumulative metrics
        self.total_posts_created = 0
        self.total_reposts = 0
        self.total_memes_created = 0
        self.total_mutations = 0
        self.total_likes = 0
        
        # Virality tracking
        self.virality_events: List[Dict] = []
        self.top_viral_posts: List[Dict] = []
        
        # Influencer tracking
        self.influencer_history: List[Dict] = []
    
    def record_tick(self, tick: int, agents: List[UserAgent], 
                    posts: List[Post], attention_scores: np.ndarray,
                    community_detector: Optional[CommunityDetector] = None,
                    language_divergence: float = 0.0,
                    meme_stats: Optional[Dict] = None):
        """Record all metrics for a single tick."""
        self.current_tick = tick
        
        metrics = {
            "tick": tick,
            "num_agents": len(agents),
            "num_posts": len(posts),
            "num_active_posts": sum(1 for p in posts if p.attention_score > 0.01),
        }
        
        # Attention metrics
        if len(attention_scores) > 0:
            metrics["mean_attention"] = float(np.mean(attention_scores))
            metrics["max_attention"] = float(np.max(attention_scores))
            metrics["attention_gini"] = self._compute_gini(attention_scores)
            metrics["attention_entropy"] = self._compute_entropy(attention_scores)
        
        # Agent metrics
        if agents:
            follower_counts = [len(a.followers) for a in agents]
            metrics["mean_followers"] = float(np.mean(follower_counts))
            metrics["follower_gini"] = self._compute_gini(np.array(follower_counts, dtype=float))
            metrics["influencer_dominance"] = self._compute_influencer_dominance(agents)
            metrics["mean_emotional_state"] = float(np.mean([a.emotional_state for a in agents]))
            
            # Personality distribution
            personality_counts = defaultdict(int)
            for a in agents:
                personality_counts[a.personality.value] += 1
            metrics["personality_distribution"] = dict(personality_counts)
        
        # Post metrics
        if posts:
            engagement_rates = [p.engagement_rate for p in posts]
            metrics["mean_engagement_rate"] = float(np.mean(engagement_rates))
            
            views = [p.views for p in posts]
            metrics["total_views"] = sum(views)
            metrics["mean_views"] = float(np.mean(views))
            
            likes = [p.likes for p in posts]
            metrics["total_likes"] = sum(likes)
            metrics["mean_likes"] = float(np.mean(likes))
            
            # Controversy metrics
            controversy_scores = [getattr(p, 'controversy_score', 0) for p in posts]
            metrics["mean_controversy"] = float(np.mean(controversy_scores))
            metrics["high_controversy_count"] = sum(1 for c in controversy_scores if c > 0.3)
            
            # Ratio metrics
            ratio_scores = [getattr(p, 'ratio_score', 0) for p in posts]
            metrics["mean_ratio"] = float(np.mean(ratio_scores))
            metrics["ratioed_count"] = sum(1 for r in ratio_scores if r > 2.0)
            
            # Distortion metrics
            distortion_levels = [getattr(p, 'distortion_level', 0) for p in posts]
            metrics["mean_distortion"] = float(np.mean(distortion_levels))
            metrics["distorted_count"] = sum(1 for d in distortion_levels if d > 0.1)
            
            # Bot content metrics
            bot_posts = sum(1 for p in posts if getattr(p, 'is_bot_post', False))
            metrics["bot_post_count"] = bot_posts
            metrics["bot_post_fraction"] = bot_posts / max(len(posts), 1)
        
        # Bot metrics
        bot_count = sum(1 for a in agents if getattr(a, 'is_bot', False))
        metrics["bot_agent_count"] = bot_count
        
        # Community metrics
        if community_detector:
            metrics["num_communities"] = len(community_detector.communities)
            metrics["ecosystem_polarization"] = community_detector.compute_ecosystem_polarization()
            metrics["community_sizes"] = {
                cid: len(c.members) for cid, c in community_detector.communities.items()
            }
        
        # Language metrics
        metrics["language_divergence"] = language_divergence
        
        # Meme metrics
        if meme_stats:
            metrics.update(meme_stats)
        
        self.tick_data.append(metrics)
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient (inequality measure)."""
        if len(values) == 0:
            return 0.0
        values = np.sort(np.asarray(values, dtype=float))
        n = len(values)
        index = np.arange(1, n + 1)
        total = np.sum(values)
        if total < 1e-10:
            return 0.0
        gini = (2 * np.sum(index * values) - (n + 1) * total) / (n * total)
        return float(np.clip(gini, 0, 1))
    
    def _compute_entropy(self, values: np.ndarray) -> float:
        """Compute Shannon entropy of attention distribution."""
        if len(values) == 0:
            return 0.0
        total = np.sum(values)
        if total < 1e-10:
            return 0.0
        probs = values / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    
    def _compute_influencer_dominance(self, agents: List[UserAgent]) -> float:
        """
        Compute how much of the attention is controlled by top influencers.
        Uses the share of total followers held by top 10% of agents.
        """
        if not agents:
            return 0.0
        
        follower_counts = sorted([len(a.followers) for a in agents], reverse=True)
        total_followers = sum(follower_counts)
        if total_followers == 0:
            return 0.0
        
        top_10_pct = max(1, len(agents) // 10)
        top_followers = sum(follower_counts[:top_10_pct])
        
        return top_followers / total_followers
    
    def record_virality_event(self, post: Post, tick: int, 
                               attention_score: float):
        """Record a virality event."""
        event = {
            "tick": tick,
            "post_id": post.id,
            "author_id": post.author_id,
            "peak_attention": post.peak_attention,
            "engagement_rate": post.engagement_rate,
            "age_at_viral": post.age_at(tick),
            "content_type": post.content_type.value,
        }
        self.virality_events.append(event)
        
        # Update top viral posts
        self.top_viral_posts.append(event)
        self.top_viral_posts.sort(key=lambda x: x["peak_attention"], reverse=True)
        self.top_viral_posts = self.top_viral_posts[:20]
    
    def get_time_series(self, metric: str) -> List[Tuple[int, float]]:
        """Get a time series for a specific metric."""
        series = []
        for data in self.tick_data:
            if metric in data:
                series.append((data["tick"], data[metric]))
        return series
    
    def get_summary(self) -> Dict:
        """Get current summary of all metrics."""
        if not self.tick_data:
            return {"status": "no_data"}
        
        latest = self.tick_data[-1]
        return {
            "current_tick": self.current_tick,
            "total_posts_created": self.total_posts_created,
            "total_reposts": self.total_reposts,
            "total_memes_created": self.total_memes_created,
            "total_mutations": self.total_mutations,
            "total_likes": self.total_likes,
            "latest_metrics": latest,
            "virality_events_count": len(self.virality_events),
        }
    
    def compute_meme_lifespan_distribution(self, posts: List[Post], 
                                            current_tick: int) -> Dict:
        """Compute the distribution of meme lifespans."""
        lifespans = []
        for post in posts:
            if post.content_type.value == "meme":
                lifespan = current_tick - post.tick_created
                # Use last attention as indicator of "death"
                if len(post.attention_history) > 0:
                    last_attention = post.attention_history[-1]
                    if last_attention < 0.01:  # Effectively dead
                        lifespans.append(lifespan)
        
        if not lifespans:
            return {"mean": 0, "median": 0, "std": 0, "distribution": []}
        
        return {
            "mean": float(np.mean(lifespans)),
            "median": float(np.median(lifespans)),
            "std": float(np.std(lifespans)),
            "distribution": lifespans,
        }
    
    def to_dict(self) -> dict:
        return self.get_summary()
