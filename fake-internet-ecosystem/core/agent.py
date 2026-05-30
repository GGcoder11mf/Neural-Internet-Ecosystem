"""
Agent system for the Fake Internet Ecosystem.

Contains UserAgent (AI individuals), BotAgent (automated accounts),
and PlatformAgent (the "internet").

Realism improvements:
- Like action (most common social media behavior)
- Circadian activity patterns (agents have active/sleep cycles)
- Doom-scrolling (agents can get trapped reading negative content)
- Content exhaustion (agents tire of over-saturated topics)
- Curiosity cycles (agents periodically explore new topics)
- Pile-on dynamics (toxic agents pile onto controversial content)
- Follow cascades (influencer follows trigger cascading follows)
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Tuple
import uuid


class PersonalityType(Enum):
    CURIOUS = "curious"           # Explores diverse content, high novelty seeking
    TOXIC = "toxic"               # Provokes, argues, spreads negativity
    VIRAL_CHASER = "viral_chaser" # Prioritizes trending/high-engagement content
    NICHE_THINKER = "niche_thinker" # Deep engagement in narrow topics
    LURKER = "lurker"             # Reads much, rarely posts
    INFLUENCER = "influencer"     # High follower base, shapes trends
    MEME_LORD = "meme_lord"       # Creates and remixes memes frequently
    ECHO_SEEKER = "echo_seeker"   # Seeks confirmation bias, follows similar views
    DOOM_SCROLLER = "doom_scroller" # Gets trapped in negative content spirals
    HOT_TAKER = "hot_taker"       # Always has provocative takes, loves controversy


# Personality behavior modifiers — heavily tuned for realism
PERSONALITY_PARAMS = {
    PersonalityType.CURIOUS: {
        "novelty_weight": 0.8, "engagement_weight": 0.3, "echo_weight": 0.1,
        "post_probability": 0.25, "repost_probability": 0.12, "like_probability": 0.35,
        "mutation_probability": 0.15, "reply_probability": 0.1,
        "attention_budget": 18, "explore_rate": 0.7,
        "doom_scroll_susceptibility": 0.15, "controversy_attraction": 0.2,
        "emoji_usage": 0.15, "hashtag_usage": 0.25,
        "avg_post_length": 35,  # words
    },
    PersonalityType.TOXIC: {
        "novelty_weight": 0.3, "engagement_weight": 0.8, "echo_weight": 0.2,
        "post_probability": 0.4, "repost_probability": 0.25, "like_probability": 0.15,
        "mutation_probability": 0.3, "reply_probability": 0.45,
        "attention_budget": 14, "explore_rate": 0.2,
        "doom_scroll_susceptibility": 0.6, "controversy_attraction": 0.9,
        "emoji_usage": 0.3, "hashtag_usage": 0.15,
        "avg_post_length": 20,
    },
    PersonalityType.VIRAL_CHASER: {
        "novelty_weight": 0.2, "engagement_weight": 0.9, "echo_weight": 0.15,
        "post_probability": 0.35, "repost_probability": 0.4, "like_probability": 0.25,
        "mutation_probability": 0.1, "reply_probability": 0.08,
        "attention_budget": 12, "explore_rate": 0.15,
        "doom_scroll_susceptibility": 0.3, "controversy_attraction": 0.6,
        "emoji_usage": 0.35, "hashtag_usage": 0.5,
        "avg_post_length": 15,
    },
    PersonalityType.NICHE_THINKER: {
        "novelty_weight": 0.5, "engagement_weight": 0.3, "echo_weight": 0.6,
        "post_probability": 0.15, "repost_probability": 0.1, "like_probability": 0.4,
        "mutation_probability": 0.08, "reply_probability": 0.2,
        "attention_budget": 10, "explore_rate": 0.3,
        "doom_scroll_susceptibility": 0.1, "controversy_attraction": 0.15,
        "emoji_usage": 0.05, "hashtag_usage": 0.35,
        "avg_post_length": 55,
    },
    PersonalityType.LURKER: {
        "novelty_weight": 0.5, "engagement_weight": 0.2, "echo_weight": 0.2,
        "post_probability": 0.03, "repost_probability": 0.03, "like_probability": 0.5,
        "mutation_probability": 0.02, "reply_probability": 0.02,
        "attention_budget": 25, "explore_rate": 0.5,
        "doom_scroll_susceptibility": 0.45, "controversy_attraction": 0.3,
        "emoji_usage": 0.08, "hashtag_usage": 0.1,
        "avg_post_length": 12,
    },
    PersonalityType.INFLUENCER: {
        "novelty_weight": 0.4, "engagement_weight": 0.7, "echo_weight": 0.2,
        "post_probability": 0.5, "repost_probability": 0.15, "like_probability": 0.1,
        "mutation_probability": 0.12, "reply_probability": 0.15,
        "attention_budget": 8, "explore_rate": 0.35,
        "doom_scroll_susceptibility": 0.2, "controversy_attraction": 0.5,
        "emoji_usage": 0.45, "hashtag_usage": 0.65,
        "avg_post_length": 25,
    },
    PersonalityType.MEME_LORD: {
        "novelty_weight": 0.7, "engagement_weight": 0.5, "echo_weight": 0.15,
        "post_probability": 0.4, "repost_probability": 0.45, "like_probability": 0.3,
        "mutation_probability": 0.6, "reply_probability": 0.15,
        "attention_budget": 14, "explore_rate": 0.5,
        "doom_scroll_susceptibility": 0.2, "controversy_attraction": 0.35,
        "emoji_usage": 0.6, "hashtag_usage": 0.4,
        "avg_post_length": 10,
    },
    PersonalityType.ECHO_SEEKER: {
        "novelty_weight": 0.1, "engagement_weight": 0.4, "echo_weight": 0.9,
        "post_probability": 0.2, "repost_probability": 0.3, "like_probability": 0.5,
        "mutation_probability": 0.03, "reply_probability": 0.1,
        "attention_budget": 12, "explore_rate": 0.05,
        "doom_scroll_susceptibility": 0.35, "controversy_attraction": 0.4,
        "emoji_usage": 0.1, "hashtag_usage": 0.2,
        "avg_post_length": 18,
    },
    PersonalityType.DOOM_SCROLLER: {
        "novelty_weight": 0.3, "engagement_weight": 0.6, "echo_weight": 0.3,
        "post_probability": 0.08, "repost_probability": 0.15, "like_probability": 0.2,
        "mutation_probability": 0.05, "reply_probability": 0.05,
        "attention_budget": 30, "explore_rate": 0.1,
        "doom_scroll_susceptibility": 0.9, "controversy_attraction": 0.7,
        "emoji_usage": 0.05, "hashtag_usage": 0.1,
        "avg_post_length": 15,
    },
    PersonalityType.HOT_TAKER: {
        "novelty_weight": 0.4, "engagement_weight": 0.8, "echo_weight": 0.15,
        "post_probability": 0.45, "repost_probability": 0.2, "like_probability": 0.08,
        "mutation_probability": 0.15, "reply_probability": 0.5,
        "attention_budget": 10, "explore_rate": 0.25,
        "doom_scroll_susceptibility": 0.3, "controversy_attraction": 0.95,
        "emoji_usage": 0.25, "hashtag_usage": 0.55,
        "avg_post_length": 28,
    },
}


class UserAgent:
    """An AI individual in the fake internet ecosystem."""
    
    def __init__(self, agent_id: str = None, interests: np.ndarray = None,
                 personality: PersonalityType = PersonalityType.CURIOUS,
                 n_topics: int = 20, name: str = None):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.name = name or f"User{np.random.randint(100,9999)}"
        self.personality = personality
        self.n_topics = n_topics
        
        # Interest vector (what topics they care about)
        if interests is not None:
            self.interests = interests.copy()
        else:
            self.interests = np.random.dirichlet(np.ones(n_topics) * 0.5)
        
        # Personality parameters
        self.params = PERSONALITY_PARAMS[personality].copy()
        
        # Memory: posts they've seen with emotional scores
        self.memory: List[Tuple[str, float]] = []  # (post_id, emotional_score)
        self.max_memory = 200
        
        # Social graph
        self.following: Set[str] = set()
        self.followers: Set[str] = set()
        
        # Community
        self.community_id: Optional[int] = None
        
        # Statistics
        self.posts_created = 0
        self.posts_reposted = 0
        self.posts_liked = 0
        self.posts_replied = 0
        self.posts_ignored = 0
        self.total_views = 0
        self.total_likes_received = 0
        
        # Emotional state (influences behavior)
        self.emotional_state = np.random.uniform(-0.3, 0.3)
        
        # Circadian pattern: each agent has an active phase
        # Phase is 0-1 representing their "timezone" offset
        self.activity_phase = np.random.uniform(0, 1)
        # Activity amplitude: how much their activity varies
        self.activity_amplitude = np.random.uniform(0.3, 0.8)
        
        # Doom-scrolling state
        self.doom_scroll_depth = 0  # How deep in a doom scroll
        self.doom_scroll_cooldown = 0
        
        # Content exhaustion: tracks topic fatigue
        self.topic_exposure: Dict[int, float] = {}  # topic_idx -> recent_exposure
        self.exhaustion_decay = 0.9
        
        # Curiosity cycle: periodic desire to explore new topics
        self.curiosity_cycle_phase = np.random.uniform(0, 2 * np.pi)
        self.curiosity_cycle_freq = np.random.uniform(0.05, 0.15)
        
        # Pile-on tracking: toxic agents remember what they've piled onto
        self.pile_on_targets: Set[str] = set()
        
        # Vocabulary (for language drift)
        self.vocabulary: Dict[str, int] = {}  # word -> frequency
        
        # Follow decision tracking
        self.last_follow_tick = 0
        self.last_unfollow_tick = 0
        
        # Is this a bot?
        self.is_bot = False
    
    def get_activity_level(self, tick: int) -> float:
        """
        Compute current activity level based on circadian rhythm.
        Returns 0.0 to 1.0 — how active this agent is right now.
        """
        # Simulate a 24-tick "day" cycle
        day_phase = (tick * 0.1 + self.activity_phase * 2 * np.pi) % (2 * np.pi)
        # Sine wave: peaks during "day", dips during "night"
        base_activity = 0.5 + self.activity_amplitude * 0.5 * np.cos(day_phase)
        # Add some noise
        base_activity += np.random.uniform(-0.05, 0.05)
        return np.clip(base_activity, 0.05, 1.0)
    
    def compute_emotional_score(self, post) -> float:
        """Compute emotional reaction to a post based on interests and personality."""
        # Interest alignment
        interest_alignment = np.dot(self.interests, post.topic_vector)
        interest_alignment = (interest_alignment + 1) / 2  # Normalize to [0, 1]
        
        # Novelty reaction
        novelty_reaction = self.params["novelty_weight"] * post.novelty_score
        
        # Controversy attraction
        controversy_score = getattr(post, 'controversy_score', 0)
        controversy_pull = self.params["controversy_attraction"] * controversy_score
        
        # Content exhaustion: less emotional response to over-exposed topics
        dominant_topic = np.argmax(post.topic_vector)
        exhaustion = self.topic_exposure.get(dominant_topic, 0)
        exhaustion_factor = max(0.2, 1.0 - exhaustion * 0.3)
        
        # Personality-specific emotional modifiers
        if self.personality == PersonalityType.TOXIC:
            # Toxic agents react more to controversial/engaging content
            emotional = (interest_alignment * 0.2 + novelty_reaction * 0.1 + 
                        controversy_pull * 0.4 + np.random.uniform(-0.4, 0.6))
        elif self.personality == PersonalityType.ECHO_SEEKER:
            # Echo seekers feel strongly about aligned content
            emotional = (interest_alignment * 0.7 + novelty_reaction * 0.1 + 
                        controversy_pull * 0.1 + np.random.uniform(-0.1, 0.1))
        elif self.personality == PersonalityType.DOOM_SCROLLER:
            # Doom scrollers are drawn to negative content
            neg_bias = -0.2 if post.engagement_rate > 0.3 else 0
            emotional = (interest_alignment * 0.3 + novelty_reaction * 0.1 + 
                        controversy_pull * 0.3 + neg_bias + np.random.uniform(-0.2, 0.2))
        elif self.personality == PersonalityType.HOT_TAKER:
            # Hot takers love controversy
            emotional = (interest_alignment * 0.2 + novelty_reaction * 0.1 + 
                        controversy_pull * 0.6 + np.random.uniform(-0.1, 0.3))
        else:
            emotional = (interest_alignment * 0.4 + novelty_reaction * 0.2 + 
                        controversy_pull * 0.15 + np.random.uniform(-0.15, 0.15))
        
        # Doom-scrolling amplification: if deep in doom scroll, negative reactions intensify
        if self.doom_scroll_depth > 3:
            emotional -= 0.1 * min(self.doom_scroll_depth, 5)
        
        # Modify by current emotional state (mood persistence)
        emotional = emotional * 0.75 + self.emotional_state * 0.25
        
        # Apply exhaustion
        emotional *= exhaustion_factor
        
        return np.clip(emotional, -1.0, 1.0)
    
    def decide_action(self, post, tick: int) -> str:
        """
        Decide what to do with a post: 'read', 'ignore', 'like', 'repost', 'remix', 'reply'.
        
        Based on personality, emotional reaction, attention budget, and activity level.
        This is the core decision tree — heavily tuned for realistic behavior patterns.
        """
        emotional_score = self.compute_emotional_score(post)
        activity_level = self.get_activity_level(tick)
        
        # Add to memory
        self.memory.append((post.id, emotional_score))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        
        # Update emotional state based on reaction
        self.emotional_state = self.emotional_state * 0.92 + emotional_score * 0.08
        
        # Update topic exposure
        for i in range(len(post.topic_vector)):
            if post.topic_vector[i] > 0.1:
                self.topic_exposure[i] = self.topic_exposure.get(i, 0) + post.topic_vector[i] * 0.1
        
        # Check if already seen this post
        seen_ids = {m[0] for m in self.memory[-30:]}
        if post.id in seen_ids and len([m for m in self.memory[-5:] if m[0] == post.id]) > 1:
            self.posts_ignored += 1
            return "ignore"
        
        # Decision logic based on emotional score
        # Very negative or very low interest = likely ignore
        if emotional_score < -0.3:
            # BUT: doom scrollers and toxic agents might still engage with negative content
            if self.params["doom_scroll_susceptibility"] > 0.5 and np.random.random() < self.params["doom_scroll_susceptibility"]:
                pass  # They engage with negative content
            elif np.random.random() > 0.15:
                self.posts_ignored += 1
                return "ignore"
        
        # Slightly positive or neutral = might still engage (lower threshold now)
        if emotional_score < 0.02:
            if np.random.random() > 0.5:  # 50% skip boring content (was 60%)
                self.posts_ignored += 1
                return "ignore"
        
        self.total_views += 1
        
        # Doom-scrolling check: negative content can trap agents
        if emotional_score < -0.1 and np.random.random() < self.params["doom_scroll_susceptibility"]:
            self.doom_scroll_depth += 1
            self.doom_scroll_cooldown = 5
        else:
            if self.doom_scroll_cooldown > 0:
                self.doom_scroll_cooldown -= 1
            else:
                self.doom_scroll_depth = max(0, self.doom_scroll_depth - 1)
        
        # === ACTION DECISION ===
        # This is a priority-based decision tree
        r = np.random.random()
        
        # Scale probabilities by activity level (less active = less engagement)
        activity_scale = 0.5 + 0.5 * activity_level
        
        # 1. MEME LORDS: remix before anything else
        if self.personality == PersonalityType.MEME_LORD and r < self.params["mutation_probability"] * activity_scale:
            self.posts_reposted += 1
            return "remix"
        
        # 2. PILE-ON: toxic/hot_taker agents pile onto controversial posts
        controversy = getattr(post, 'controversy_score', 0)
        if controversy > 0.5 and self.personality in (PersonalityType.TOXIC, PersonalityType.HOT_TAKER):
            if r < self.params["reply_probability"] * controversy * activity_scale:
                self.pile_on_targets.add(post.id)
                self.posts_replied += 1
                return "reply"
        
        # 3. LIKE: the most common action by far (but was completely missing!)
        like_prob = self.params["like_probability"] * max(emotional_score * 0.8, 0.05) * activity_scale
        # Echo seekers like content that aligns with their views
        if self.personality == PersonalityType.ECHO_SEEKER:
            interest_alignment = np.dot(self.interests, post.topic_vector)
            like_prob *= (1 + interest_alignment)
        # Lurkers like more than they post
        if self.personality == PersonalityType.LURKER:
            like_prob *= 1.5
        
        if r < like_prob:
            self.posts_liked += 1
            return "like"
        
        # 4. REPLY: driven by emotional intensity and personality
        reply_prob = self.params["reply_probability"] * max(abs(emotional_score), 0.1) * activity_scale
        # Controversy drives replies more
        if controversy > 0.3:
            reply_prob *= (1 + controversy * 0.5)
        
        if r < like_prob + reply_prob:
            self.posts_replied += 1
            return "reply"
        
        # 5. REPOST: driven by desire to share
        repost_prob = self.params["repost_probability"] * max(emotional_score, 0.1) * activity_scale
        # Viral chasers repost trending content more
        if self.personality == PersonalityType.VIRAL_CHASER and post.attention_score > 2.0:
            repost_prob *= 2.0
        
        if r < like_prob + reply_prob + repost_prob:
            self.posts_reposted += 1
            return "repost"
        
        # 6. Default: just read
        return "read"
    
    def should_post(self, tick: int) -> bool:
        """Decide whether to create a new post this tick."""
        activity_level = self.get_activity_level(tick)
        
        # Scale posting probability by activity level
        r = np.random.random()
        base_prob = self.params["post_probability"] * activity_level
        
        # Emotional boost: extreme emotions drive posting
        emotional_boost = abs(self.emotional_state) * 0.08
        
        # Curiosity cycle: agents in exploration phase post more
        curiosity = 0.5 + 0.5 * np.sin(tick * self.curiosity_cycle_freq + self.curiosity_cycle_phase)
        curiosity_boost = curiosity * 0.05
        
        return r < base_prob + emotional_boost + curiosity_boost
    
    def should_follow(self, other_agent_id: str, similarity: float, tick: int,
                      other_is_influencer: bool = False) -> bool:
        """Decide whether to follow another agent."""
        if other_agent_id in self.following:
            return False
        if tick - self.last_follow_tick < 5:
            return False
        
        # Base threshold
        follow_threshold = 0.3 + (1 - self.params["echo_weight"]) * 0.4
        
        # Echo seekers follow similar agents more readily
        if self.personality == PersonalityType.ECHO_SEEKER:
            follow_threshold -= 0.15
        
        if similarity > follow_threshold and np.random.random() < 0.08:
            self.last_follow_tick = tick
            return True
        
        # Random follow (like how real people follow sometimes)
        if np.random.random() < 0.005:
            self.last_follow_tick = tick
            return True
        
        return False
    
    def should_unfollow(self, other_agent_id: str, similarity: float, tick: int) -> bool:
        """Decide whether to unfollow an agent."""
        if other_agent_id not in self.following:
            return False
        if tick - self.last_unfollow_tick < 10:
            return False
        
        # Unfollow if interests diverged significantly
        if similarity < 0.1 and np.random.random() < 0.03:
            self.last_unfollow_tick = tick
            return True
        
        # Random unfollow (people clean their follows sometimes)
        if np.random.random() < 0.002:
            self.last_unfollow_tick = tick
            return True
        
        return False
    
    def update_interests(self, post, learning_rate: float = 0.015):
        """Slowly shift interests based on consumed content."""
        self.interests = self.interests * (1 - learning_rate) + post.topic_vector * learning_rate
        # Re-normalize
        self.interests = np.clip(self.interests, 0.01, None)
        self.interests /= self.interests.sum()
    
    def decay_exhaustion(self):
        """Decay topic exposure over time (agents recover from content fatigue)."""
        for topic in list(self.topic_exposure.keys()):
            self.topic_exposure[topic] *= self.exhaustion_decay
            if self.topic_exposure[topic] < 0.01:
                del self.topic_exposure[topic]
    
    def to_dict(self) -> dict:
        """Serialize agent state to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "personality": self.personality.value,
            "interests": self.interests.tolist(),
            "following": list(self.following),
            "followers": list(self.followers),
            "community_id": self.community_id,
            "posts_created": self.posts_created,
            "posts_reposted": self.posts_reposted,
            "posts_liked": self.posts_liked,
            "posts_replied": self.posts_replied,
            "posts_ignored": self.posts_ignored,
            "total_views": self.total_views,
            "total_likes_received": self.total_likes_received,
            "emotional_state": round(self.emotional_state, 3),
            "doom_scroll_depth": self.doom_scroll_depth,
            "activity_level": round(self.get_activity_level(0), 2),
            "is_bot": self.is_bot,
        }


class BotAgent(UserAgent):
    """
    A bot account that amplifies, spams, or coordinates inauthentic behavior.
    
    Types:
    - amplifier: Retweets specific content to boost visibility
    - spammer: Posts repetitive content (ads, scams, engagement bait)
    - coordinator: Part of a coordinated network that amplifies together
    """
    
    BOT_TYPES = ["amplifier", "spammer", "coordinator"]
    
    def __init__(self, bot_type: str = "amplifier", target_topics: np.ndarray = None,
                 coordinator_group: int = 0, **kwargs):
        # Bots are typically echo_seekers or viral_chasers
        default_personality = PersonalityType.VIRAL_CHASER
        if bot_type == "spammer":
            default_personality = PersonalityType.INFLUENCER
        
        super().__init__(personality=default_personality, **kwargs)
        
        self.bot_type = bot_type
        self.is_bot = True
        self.coordinator_group = coordinator_group
        
        # Target topics to amplify (for amplifier bots)
        self.target_topics = target_topics
        if self.target_topics is None:
            # Random target topics
            self.target_topics = np.zeros(self.n_topics)
            n_targets = np.random.randint(1, 4)
            targets = np.random.choice(self.n_topics, n_targets, replace=False)
            self.target_topics[targets] = np.random.uniform(0.3, 1.0, n_targets)
        
        # Override some parameters for bot behavior
        self.name = self._generate_bot_name()
        
        # Bot-specific parameters
        self.spam_interval = np.random.randint(3, 8)  # ticks between spam posts
        self.amplification_boost = np.random.uniform(2.0, 5.0)  # how much they boost
        self.coordination_delay = np.random.randint(0, 3)  # staggered amplification
    
    def _generate_bot_name(self) -> str:
        """Generate a realistic-looking bot name."""
        prefixes = ["crypto", "news", "daily", "update", "alert", "trending", 
                     "insider", "truth", "source", "watch", "now", "live"]
        suffixes = ["bot", "2024", "news", "daily", "hq", "feed", "net", 
                     "wire", "hq", "central", "zone", "hub"]
        style = np.random.randint(0, 3)
        if style == 0:
            return f"{np.random.choice(prefixes)}_{np.random.choice(suffixes)}"
        elif style == 1:
            return f"{np.random.choice(prefixes).title()}{np.random.choice(suffixes).title()}"
        else:
            return f"{np.random.choice(prefixes)}{np.random.randint(10,9999)}"
    
    def decide_action(self, post, tick: int) -> str:
        """Bot decision logic — different from human agents."""
        if self.bot_type == "amplifier":
            # Amplify content matching target topics
            topic_match = np.dot(post.topic_vector, self.target_topics)
            if topic_match > 0.3 and post.attention_score > 0.5:
                if np.random.random() < 0.6:  # High repost rate
                    self.posts_reposted += 1
                    return "repost"
                elif np.random.random() < 0.3:
                    self.posts_liked += 1
                    return "like"
            return "ignore"
        
        elif self.bot_type == "coordinator":
            # Amplify in coordinated waves
            if tick % 5 == self.coordination_delay:
                topic_match = np.dot(post.topic_vector, self.target_topics)
                if topic_match > 0.2:
                    self.posts_reposted += 1
                    return "repost"
            return "ignore"
        
        elif self.bot_type == "spammer":
            # Rarely interact with others' content
            if np.random.random() < 0.05:
                self.posts_liked += 1
                return "like"
            return "ignore"
        
        return "ignore"
    
    def should_post(self, tick: int) -> bool:
        """Bot posting logic."""
        if self.bot_type == "spammer":
            return tick % self.spam_interval == 0
        elif self.bot_type == "amplifier":
            # Post occasionally to look legit
            return np.random.random() < 0.08
        elif self.bot_type == "coordinator":
            # Post on coordination schedule
            return tick % 8 == self.coordination_delay and np.random.random() < 0.3
        return False


class PlatformAgent:
    """The platform/internet agent that controls feed ranking, trending, and recommendations."""
    
    def __init__(self, n_topics: int = 20, echo_chamber_strength: float = 0.5,
                 virality_sensitivity: float = 1.0, novelty_decay: float = 0.95,
                 recommendation_bias: float = 0.3, controversy_boost: float = 0.4):
        self.n_topics = n_topics
        self.echo_chamber_strength = echo_chamber_strength
        self.virality_sensitivity = virality_sensitivity
        self.novelty_decay = novelty_decay
        self.recommendation_bias = recommendation_bias
        self.controversy_boost = controversy_boost
        
        # Trending posts
        self.trending: List[str] = []
        self.trending_threshold = 5.0
        
        # Platform-level content moderation / boosting
        self.boosted_topics = np.zeros(n_topics)
        self.suppressed_topics = np.zeros(n_topics)
        
        # Algorithm evolution
        self.algorithm_mutation_rate = 0.001
        self.tick_count = 0
    
    def compute_platform_bias(self, posts: list) -> np.ndarray:
        """Compute platform bias scores for all posts."""
        bias_scores = np.zeros(len(posts))
        for i, post in enumerate(posts):
            # Boost trending content
            if post.id in self.trending:
                bias_scores[i] += 0.5
            
            # Apply topic-level boost/suppress
            topic_boost = np.dot(post.topic_vector, self.boosted_topics)
            topic_suppress = np.dot(post.topic_vector, self.suppressed_topics)
            bias_scores[i] += topic_boost - topic_suppress * 0.5
            
            # Recommendation bias: prefer content with existing engagement
            engagement = post.likes + post.reposts + post.replies
            bias_scores[i] += np.log1p(engagement) * self.recommendation_bias
            
            # Controversy boost: platform loves controversial content (engagement!)
            controversy = getattr(post, 'controversy_score', 0)
            bias_scores[i] += controversy * self.controversy_boost
        
        return bias_scores
    
    def update_trending(self, posts: list, attention_scores: np.ndarray):
        """Update trending list based on attention scores."""
        self.trending = []
        for i, post in enumerate(posts):
            if attention_scores[i] > self.trending_threshold:
                self.trending.append(post.id)
        
        # Limit trending to top 20
        if len(self.trending) > 20:
            post_attention = {p.id: attention_scores[i] for i, p in enumerate(posts) if p.id in self.trending}
            self.trending = sorted(post_attention, key=post_attention.get, reverse=True)[:20]
    
    def evolve_algorithm(self):
        """Slowly mutate platform algorithm parameters."""
        self.tick_count += 1
        
        if np.random.random() < self.algorithm_mutation_rate:
            self.echo_chamber_strength += np.random.uniform(-0.05, 0.05)
            self.echo_chamber_strength = np.clip(self.echo_chamber_strength, 0.0, 1.0)
        
        if np.random.random() < self.algorithm_mutation_rate:
            self.virality_sensitivity += np.random.uniform(-0.05, 0.05)
            self.virality_sensitivity = np.clip(self.virality_sensitivity, 0.1, 3.0)
        
        # Controversy boost slowly increases (platforms optimize for engagement)
        if np.random.random() < 0.005:
            self.controversy_boost += np.random.uniform(0, 0.02)
            self.controversy_boost = np.clip(self.controversy_boost, 0, 1.0)
        
        # Slowly shift boosted topics
        if self.tick_count % 50 == 0:
            self.boosted_topics = np.random.dirichlet(np.ones(self.n_topics) * 0.1) * 0.2
    
    def to_dict(self) -> dict:
        """Serialize platform state."""
        return {
            "echo_chamber_strength": round(self.echo_chamber_strength, 3),
            "virality_sensitivity": round(self.virality_sensitivity, 3),
            "novelty_decay": round(self.novelty_decay, 3),
            "recommendation_bias": round(self.recommendation_bias, 3),
            "controversy_boost": round(self.controversy_boost, 3),
            "trending_count": len(self.trending),
            "trending": self.trending,
            "tick_count": self.tick_count,
        }
