"""
Main Simulation Loop for the Fake Internet Ecosystem.

Each tick = one "moment" in internet time:
1. Agents receive feed (ranked by platform + neural net)
2. They decide: read, ignore, like, repost, remix content, reply
3. New content is generated (including bot spam)
4. Attention scores update
5. Controversy scores update
6. Virality spreads or dies
7. Information gets distorted through reposts (telephone game)
8. Follow cascades happen
9. Sentiment waves emerge
10. Neural network trains on new data
11. Meme images are generated

v3 improvements:
- PyTorch CUDA neural network for feed ranking
- Actual meme image generation (PIL)
- Real-time tick support
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .agent import UserAgent, BotAgent, PlatformAgent, PersonalityType
from .content import (Post, Meme, Idea, ContentType, generate_post_text, 
                       generate_hashtags, distort_repost_text, TOPIC_LABELS)
from .attention import AttentionEngine
from .meme_evolution import MemeEvolver
from .community import CommunityDetector
from .language_drift import LanguageDriftEngine
from .metrics import MetricsTracker
from .meme_generator import generate_meme_image, generate_post_image, cleanup_old_images
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.backend import compute_attention_matrix, compute_similarity_matrix, rank_feed
from gpu.neural_net import NeuralRankingEngine, TORCH_AVAILABLE


class Simulation:
    """
    The main simulation engine orchestrating the entire fake internet ecosystem.
    """
    
    def __init__(self, n_agents: int = 50, n_topics: int = 20,
                 echo_chamber_strength: float = 0.5,
                 virality_sensitivity: float = 1.0,
                 novelty_decay: float = 0.95,
                 mutation_rate: float = 0.3,
                 community_detection_interval: int = 10,
                 max_posts: int = 500,
                 bot_fraction: float = 0.1,
                 sentiment_wave_probability: float = 0.02,
                 generate_images: bool = True):
        
        self.n_agents = n_agents
        self.n_topics = n_topics
        self.max_posts = max_posts
        self.community_detection_interval = community_detection_interval
        self.bot_fraction = bot_fraction
        self.sentiment_wave_probability = sentiment_wave_probability
        self.generate_images = generate_images
        
        # Initialize subsystems
        self.platform = PlatformAgent(
            n_topics=n_topics,
            echo_chamber_strength=echo_chamber_strength,
            virality_sensitivity=virality_sensitivity,
            novelty_decay=novelty_decay
        )
        
        self.attention_engine = AttentionEngine(
            novelty_decay=novelty_decay,
            virality_threshold=5.0
        )
        
        self.meme_evolver = MemeEvolver(
            mutation_rate=mutation_rate,
            merge_probability=0.1,
            selection_pressure=0.5
        )
        
        self.community_detector = CommunityDetector(
            n_topics=n_topics,
            min_community_size=3
        )
        
        self.language_engine = LanguageDriftEngine(
            compression_rate=0.1,
            slang_generation_rate=0.05,
            emoji_replacement_rate=0.08
        )
        
        self.metrics = MetricsTracker()
        
        # Neural ranking engine
        self.neural_ranker = NeuralRankingEngine(n_topics=n_topics)
        
        # State
        self.agents: List[UserAgent] = []
        self.agent_map: Dict[str, UserAgent] = {}
        self.posts: List[Post] = []
        self.post_map: Dict[str, Post] = {}
        self.tick = 0
        self.running = False
        
        # Sentiment wave tracking
        self.active_sentiment_wave: Optional[Dict] = None
        self.sentiment_wave_history: List[Dict] = []
        
        # Follow cascade tracking
        self.follow_cascades: List[Dict] = []
        
        # Image generation counter (limit images per tick)
        self.images_this_tick = 0
        self.max_images_per_tick = 5
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Create the initial population of agents (including bots)."""
        personality_weights = {
            PersonalityType.CURIOUS: 0.15,
            PersonalityType.TOXIC: 0.08,
            PersonalityType.VIRAL_CHASER: 0.12,
            PersonalityType.NICHE_THINKER: 0.12,
            PersonalityType.LURKER: 0.12,
            PersonalityType.INFLUENCER: 0.04,
            PersonalityType.MEME_LORD: 0.08,
            PersonalityType.ECHO_SEEKER: 0.1,
            PersonalityType.DOOM_SCROLLER: 0.1,
            PersonalityType.HOT_TAKER: 0.09,
        }
        
        personalities = list(personality_weights.keys())
        weights = list(personality_weights.values())
        
        n_bots = max(1, int(self.n_agents * self.bot_fraction))
        n_humans = self.n_agents - n_bots
        
        # Create human agents
        for i in range(n_humans):
            personality = np.random.choice(personalities, p=weights)
            
            # Generate interest vector with some clustering
            n_dominant = np.random.randint(2, 5)
            dominant_topics = np.random.choice(self.n_topics, n_dominant, replace=False)
            interests = np.random.dirichlet(np.ones(self.n_topics) * 0.1)
            interests[dominant_topics] += np.random.uniform(0.1, 0.3, n_dominant)
            interests /= interests.sum()
            
            agent = UserAgent(
                agent_id=f"a{i:04d}",
                interests=interests,
                personality=personality,
                n_topics=self.n_topics,
            )
            
            self.agents.append(agent)
            self.agent_map[agent.id] = agent
        
        # Create bot agents
        for i in range(n_bots):
            bot_type = np.random.choice(["amplifier", "spammer", "coordinator"],
                                         p=[0.4, 0.3, 0.3])
            
            # Bots have target topics
            target_topics = np.zeros(self.n_topics)
            n_targets = np.random.randint(1, 4)
            targets = np.random.choice(self.n_topics, n_targets, replace=False)
            target_topics[targets] = np.random.uniform(0.3, 1.0, n_targets)
            
            interests = np.random.dirichlet(np.ones(self.n_topics) * 0.1)
            interests[targets] += np.random.uniform(0.1, 0.3, n_targets)
            interests /= interests.sum()
            
            bot = BotAgent(
                agent_id=f"bot{i:04d}",
                bot_type=bot_type,
                target_topics=target_topics,
                coordinator_group=i % 3,
                interests=interests,
                n_topics=self.n_topics,
            )
            
            self.agents.append(bot)
            self.agent_map[bot.id] = bot
        
        # Create initial social connections
        self._initialize_social_graph()
        
        # Seed initial content
        self._seed_initial_content()
    
    def _initialize_social_graph(self):
        """Create initial following relationships."""
        for agent in self.agents:
            # Each agent follows 3-8 random agents
            n_follow = np.random.randint(3, 9)
            others = [a for a in self.agents if a.id != agent.id]
            if others:
                n_follow = min(n_follow, len(others))
                # Prefer similar interests
                similarities = [np.dot(agent.interests, o.interests) for o in others]
                probs = np.array(similarities)
                probs = np.maximum(probs, 0.01)
                probs /= probs.sum()
                
                selected = np.random.choice(len(others), n_follow, replace=False, p=probs)
                for idx in selected:
                    other = others[idx]
                    agent.following.add(other.id)
                    other.followers.add(agent.id)
        
        # Influencers start with more followers
        for agent in self.agents:
            if agent.personality == PersonalityType.INFLUENCER:
                # Add random followers
                potential = [a for a in self.agents if a.id != agent.id and a.id not in agent.followers]
                n_extra = min(len(potential), np.random.randint(5, 15))
                for other in np.random.choice(len(potential), n_extra, replace=False):
                    follower = potential[other]
                    follower.following.add(agent.id)
                    agent.followers.add(follower.id)
    
    def _seed_initial_content(self):
        """Seed the ecosystem with initial posts so agents have content to interact with."""
        self.images_this_tick = 0  # Allow more seed images
        for agent in self.agents:
            n_posts = np.random.randint(1, 3)
            for _ in range(n_posts):
                post = self._create_post(agent)
                if post:
                    post.tick_created = 0
                    self.posts.append(post)
                    self.post_map[post.id] = post
                    self.metrics.total_posts_created += 1
                    if isinstance(post, Meme):
                        self.metrics.total_memes_created += 1
    
    def run_tick(self) -> Dict:
        """Execute one tick of the simulation."""
        self.tick += 1
        self.images_this_tick = 0
        
        # Step 1: Maybe trigger a sentiment wave
        self._check_sentiment_wave()
        
        # Step 2: Apply sentiment wave effects
        if self.active_sentiment_wave:
            self._apply_sentiment_wave()
            self.active_sentiment_wave["duration"] -= 1
            if self.active_sentiment_wave["duration"] <= 0:
                self.sentiment_wave_history.append(self.active_sentiment_wave)
                self.active_sentiment_wave = None
        
        # Step 3: Compute attention for all posts
        attention_scores = self._compute_attention()
        
        # Step 4: Platform updates trending and evolves algorithm
        self.platform.update_trending(self.posts, attention_scores)
        self.platform.evolve_algorithm()
        
        # Step 5: Generate feeds and process agent decisions
        new_posts = self._process_agents(attention_scores)
        
        # Step 6: Add new posts
        for post in new_posts:
            self.posts.append(post)
            self.post_map[post.id] = post
            self.metrics.total_posts_created += 1
            if isinstance(post, Meme):
                self.metrics.total_memes_created += 1
        
        # Step 7: Prune old/low-attention posts
        self._prune_posts()
        
        # Step 8: Update social graph (with follow cascades)
        self._update_social_graph()
        
        # Step 9: Decay content exhaustion for all agents
        for agent in self.agents:
            if hasattr(agent, 'decay_exhaustion'):
                agent.decay_exhaustion()
        
        # Step 10: Community detection (periodic)
        if self.tick % self.community_detection_interval == 0:
            self.community_detector.detect_communities(self.agents)
            self.community_detector.record_history(self.tick)
        
        # Step 11: Language drift
        self._apply_language_drift()
        language_divergence = self.language_engine.compute_language_divergence()
        
        # Step 12: Meme evolution stats
        meme_stats = self.meme_evolver.to_dict()
        n_original = len(attention_scores)
        original_posts = self.posts[:n_original] if n_original > 0 else []
        meme_posts_in_original = [p for p in original_posts if isinstance(p, Meme)]
        if meme_posts_in_original and len(attention_scores) > 0:
            post_attention_map = {p.id: attention_scores[i] for i, p in enumerate(original_posts)}
            meme_attention = np.array([post_attention_map[m.id] for m in meme_posts_in_original if m.id in post_attention_map])
            if len(meme_attention) > 0:
                self.meme_evolver.apply_selection_pressure(meme_posts_in_original, meme_attention)
                self.meme_evolver.update_dominant_memes(meme_posts_in_original, meme_attention)
        
        # Step 13: Train neural network
        if self.neural_ranker.should_train():
            loss = self.neural_ranker.train_step()
        
        # Step 14: Record metrics
        self.metrics.record_tick(
            tick=self.tick,
            agents=self.agents,
            posts=self.posts,
            attention_scores=attention_scores,
            community_detector=self.community_detector,
            language_divergence=language_divergence,
            meme_stats=meme_stats
        )
        
        # Record virality events
        for i, post in enumerate(self.posts):
            if i < len(attention_scores) and self.attention_engine.is_viral(attention_scores[i]):
                if post.peak_attention == attention_scores[i]:
                    self.metrics.record_virality_event(post, self.tick, attention_scores[i])
        
        # Cleanup old images periodically
        if self.tick % 50 == 0:
            cleanup_old_images()
        
        return self.metrics.tick_data[-1] if self.metrics.tick_data else {}
    
    def run_ticks(self, n: int, callback=None) -> List[Dict]:
        """Run multiple ticks."""
        results = []
        for i in range(n):
            result = self.run_tick()
            results.append(result)
            if callback and i % 10 == 0:
                callback(i, n, result)
        return results
    
    def _compute_attention(self) -> np.ndarray:
        """Compute attention scores for all posts."""
        if not self.posts:
            return np.array([])
        return self.attention_engine.compute_all_attention(self.posts, self.tick)
    
    def _process_agents(self, attention_scores: np.ndarray) -> List[Post]:
        """Process each agent's decisions for this tick."""
        new_posts = []
        
        has_posts = bool(self.posts) and len(attention_scores) > 0
        
        # Compute platform bias only if there are posts
        platform_biases = self.platform.compute_platform_bias(self.posts) if has_posts else np.array([])
        
        # Compute neural network predictions for all posts (batch)
        nn_predictions = None
        if has_posts and self.neural_ranker.model is not None:
            nn_predictions = self.neural_ranker.predict_batch(self.posts, self.agent_map)
        
        for agent in self.agents:
            # Check activity level (circadian rhythm)
            activity_level = agent.get_activity_level(self.tick)
            
            # Skip inactive agents most of the time
            if np.random.random() > activity_level:
                continue
            
            # --- Feed consumption (only if posts exist) ---
            if has_posts:
                # Compute network distances
                network_distances = np.array([
                    self.attention_engine.compute_network_distance(
                        agent, post.author_id, self.agent_map
                    ) for post in self.posts
                ])
                
                # Compute feed ranking (with NN predictions)
                post_topics = np.array([p.topic_vector for p in self.posts])
                ranked_indices, feed_scores = rank_feed(
                    attention_scores=attention_scores,
                    agent_interests=agent.interests,
                    post_topics=post_topics,
                    platform_bias=platform_biases,
                    network_distances=network_distances,
                    echo_chamber_strength=self.platform.echo_chamber_strength,
                    virality_sensitivity=self.platform.virality_sensitivity,
                    nn_predictions=nn_predictions
                )
                
                # Process top posts within attention budget
                budget = int(agent.params["attention_budget"] * activity_level)
                seen_count = 0
                
                for idx in ranked_indices[:budget * 2]:
                    if seen_count >= budget:
                        break
                    
                    idx = int(idx)
                    if idx >= len(self.posts):
                        continue
                    
                    post = self.posts[idx]
                    action = agent.decide_action(post, self.tick)
                    
                    if action == "ignore":
                        post.ignores += 1
                        continue
                    
                    seen_count += 1
                    post.views += 1
                    
                    if action == "like":
                        post.likes += 1
                        if hasattr(post, 'author_id') and post.author_id in self.agent_map:
                            self.agent_map[post.author_id].total_likes_received += 1
                        self.metrics.total_likes += 1
                        agent.update_interests(post, learning_rate=0.005)
                        
                        # Add training sample to neural net
                        engagement = post.engagement_rate
                        self.neural_ranker.add_training_sample(post, agent, engagement)
                    
                    elif action == "repost":
                        post.reposts += 1
                        self.metrics.total_reposts += 1
                        
                        # Apply information distortion (telephone game)
                        repost_text = post.text
                        distortion = 0.0
                        if np.random.random() < 0.15:
                            repost_text, distortion = distort_repost_text(post.text)
                        
                        # Maybe mutate if it's a meme
                        if isinstance(post, Meme) and self.meme_evolver.should_mutate(agent.personality.value):
                            child_meme = self.meme_evolver.evolve_meme(
                                post, agent.id, self.tick,
                                available_memes=[p for p in self.posts if isinstance(p, Meme)]
                            )
                            if distortion > 0:
                                child_meme.distortion_level = post.distortion_level + distortion
                            child_meme.text = self.language_engine.process_text(
                                child_meme.text, agent.community_id or 0, self.tick
                            )
                            child_meme.hashtags = post.hashtags
                            # Generate image for mutated meme
                            if self.generate_images and self.images_this_tick < self.max_images_per_tick:
                                self.images_this_tick += 1
                                child_meme.image_path = generate_meme_image(
                                    child_meme.text, agent.personality.value,
                                    child_meme.generation,
                                    [h["type"] for h in child_meme.mutation_history],
                                    np.argmax(child_meme.topic_vector),
                                    child_meme.id
                                )
                            new_posts.append(child_meme)
                            self.metrics.total_mutations += 1
                        else:
                            # Simple repost (with possible distortion)
                            repost = Post(
                                author_id=agent.id,
                                content_type=ContentType.TEXT,
                                text=repost_text,
                                topic_vector=post.topic_vector.copy(),
                                n_topics=self.n_topics,
                                tick_created=self.tick,
                                parent_id=post.id,
                                hashtags=post.hashtags,
                                is_bot_post=agent.is_bot,
                            )
                            repost.generation = post.generation
                            repost.distortion_level = post.distortion_level + distortion
                            repost.original_text = post.original_text
                            new_posts.append(repost)
                        
                        agent.update_interests(post)
                        # NN training sample
                        self.neural_ranker.add_training_sample(post, agent, post.engagement_rate)
                    
                    elif action == "remix":
                        post.reposts += 1
                        self.metrics.total_reposts += 1
                        
                        if isinstance(post, Meme):
                            child = self.meme_evolver.evolve_meme(
                                post, agent.id, self.tick,
                                available_memes=[p for p in self.posts if isinstance(p, Meme)]
                            )
                            child.text = self.language_engine.process_text(
                                child.text, agent.community_id or 0, self.tick
                            )
                            child.hashtags = generate_hashtags(
                                child.topic_vector, self.n_topics, agent.personality.value,
                                agent.params.get("hashtag_usage", 0.3)
                            )
                            # Generate image for remixed meme
                            if self.generate_images and self.images_this_tick < self.max_images_per_tick:
                                self.images_this_tick += 1
                                child.image_path = generate_meme_image(
                                    child.text, agent.personality.value,
                                    child.generation,
                                    [h["type"] for h in child.mutation_history],
                                    np.argmax(child.topic_vector),
                                    child.id
                                )
                            new_posts.append(child)
                            self.metrics.total_mutations += 1
                        else:
                            # Convert to meme via remix
                            meme = Meme(
                                author_id=agent.id,
                                text=post.text,
                                topic_vector=post.topic_vector.copy(),
                                n_topics=self.n_topics,
                                tick_created=self.tick,
                                parent_meme_id=post.id
                            )
                            meme.generation = 1
                            meme.text = self.language_engine.process_text(
                                meme.text, agent.community_id or 0, self.tick
                            )
                            meme.hashtags = post.hashtags
                            # Generate image for new meme
                            if self.generate_images and self.images_this_tick < self.max_images_per_tick:
                                self.images_this_tick += 1
                                meme.image_path = generate_meme_image(
                                    meme.text, agent.personality.value,
                                    meme.generation, [],
                                    np.argmax(meme.topic_vector),
                                    meme.id
                                )
                            new_posts.append(meme)
                            self.metrics.total_memes_created += 1
                        
                        agent.update_interests(post)
                        self.neural_ranker.add_training_sample(post, agent, post.engagement_rate)
                    
                    elif action == "reply":
                        post.replies += 1
                        # Replies generate new posts
                        if np.random.random() < 0.4:
                            reply_text, reply_hashtags = generate_post_text(
                                post.topic_vector * 0.6 + agent.interests * 0.4,
                                agent.personality.value,
                                agent.params.get("emoji_usage", 0.2),
                                agent.params.get("hashtag_usage", 0.3),
                                agent.is_bot,
                                self.n_topics
                            )
                            reply = Post(
                                author_id=agent.id,
                                content_type=ContentType.TEXT,
                                text=reply_text,
                                topic_vector=post.topic_vector * 0.6 + agent.interests * 0.4,
                                n_topics=self.n_topics,
                                tick_created=self.tick,
                                parent_id=post.id,
                                hashtags=reply_hashtags,
                                is_bot_post=agent.is_bot,
                            )
                            norm = np.linalg.norm(reply.topic_vector)
                            if norm > 0:
                                reply.topic_vector /= norm
                            reply.thread_id = post.id
                            reply.thread_position = post.thread_position + 1
                            new_posts.append(reply)
                        
                        agent.update_interests(post)
                    
                    elif action == "read":
                        agent.update_interests(post, learning_rate=0.008)
            
            # Maybe create original post
            if agent.should_post(self.tick):
                new_post = self._create_post(agent)
                if new_post:
                    new_posts.append(new_post)
        
        return new_posts
    
    def _create_post(self, agent: UserAgent) -> Optional[Post]:
        """Agent creates a new original post."""
        # Generate topic vector influenced by agent interests
        topic_vector = agent.interests * 0.7 + np.random.dirichlet(np.ones(self.n_topics) * 0.3) * 0.3
        norm = np.linalg.norm(topic_vector)
        if norm > 0:
            topic_vector /= norm
        
        # Generate text with personality-driven templates, emojis, hashtags
        text, hashtags = generate_post_text(
            topic_vector, 
            agent.personality.value,
            agent.params.get("emoji_usage", 0.2),
            agent.params.get("hashtag_usage", 0.3),
            agent.is_bot,
            self.n_topics
        )
        
        # Apply language drift
        community_id = agent.community_id or 0
        text = self.language_engine.process_text(text, community_id, self.tick)
        
        # Meme lords and some personalities more likely to create memes
        is_meme = False
        if agent.personality == PersonalityType.MEME_LORD and np.random.random() < 0.5:
            is_meme = True
        elif agent.personality == PersonalityType.HOT_TAKER and np.random.random() < 0.2:
            is_meme = True
        elif np.random.random() < 0.05:
            is_meme = True
        
        if is_meme:
            post = Meme(
                author_id=agent.id,
                text=text,
                topic_vector=topic_vector,
                n_topics=self.n_topics,
                tick_created=self.tick
            )
            post.hashtags = hashtags
            # Generate meme image
            if self.generate_images and self.images_this_tick < self.max_images_per_tick:
                self.images_this_tick += 1
                post.image_path = generate_meme_image(
                    text, agent.personality.value,
                    0, [], np.argmax(topic_vector), post.id
                )
        else:
            post = Post(
                author_id=agent.id,
                content_type=ContentType.TEXT,
                text=text,
                topic_vector=topic_vector,
                n_topics=self.n_topics,
                tick_created=self.tick,
                hashtags=hashtags,
                is_bot_post=agent.is_bot,
            )
            # Generate post image for some posts (not all, to save resources)
            if self.generate_images and np.random.random() < 0.3 and self.images_this_tick < self.max_images_per_tick:
                self.images_this_tick += 1
                post.image_path = generate_post_image(
                    text, agent.personality.value,
                    np.argmax(topic_vector), post.id
                )
        
        agent.posts_created += 1
        return post
    
    def _prune_posts(self):
        """Remove old/low-attention posts to keep memory bounded."""
        if len(self.posts) <= self.max_posts:
            return
        
        # Sort by attention score and keep top max_posts
        self.posts.sort(key=lambda p: p.attention_score, reverse=True)
        removed = self.posts[self.max_posts:]
        for post in removed:
            if post.id in self.post_map:
                del self.post_map[post.id]
        
        self.posts = self.posts[:self.max_posts]
    
    def _update_social_graph(self):
        """Update following/unfollowing decisions with follow cascades."""
        if self.tick % 5 != 0:
            return
        
        for agent in self.agents:
            # Consider following agents they interact with
            sample_size = min(5, len(self.agents))
            sample_indices = np.random.choice(len(self.agents), sample_size, replace=False)
            
            for idx in sample_indices:
                other = self.agents[idx]
                if other.id == agent.id:
                    continue
                
                similarity = np.dot(agent.interests, other.interests)
                other_is_influencer = (other.personality == PersonalityType.INFLUENCER)
                
                if agent.should_follow(other.id, similarity, self.tick, other_is_influencer):
                    agent.following.add(other.id)
                    other.followers.add(agent.id)
                    
                    if other_is_influencer and np.random.random() < 0.15:
                        self._trigger_follow_cascade(other, agent)
                
                if agent.should_unfollow(other.id, similarity, self.tick):
                    agent.following.discard(other.id)
                    other.followers.discard(agent.id)
    
    def _trigger_follow_cascade(self, influencer: UserAgent, followed_agent: UserAgent):
        """When an influencer follows someone, some of their followers follow too."""
        cascade_size = 0
        for follower_id in list(influencer.followers)[:10]:
            if follower_id in self.agent_map:
                follower = self.agent_map[follower_id]
                if follower_id not in followed_agent.followers:
                    if np.random.random() < 0.08:
                        follower.following.add(followed_agent.id)
                        followed_agent.followers.add(follower_id)
                        cascade_size += 1
        
        if cascade_size > 0:
            self.follow_cascades.append({
                "tick": self.tick,
                "influencer": influencer.id,
                "target": followed_agent.id,
                "cascade_size": cascade_size,
            })
    
    def _check_sentiment_wave(self):
        """Maybe trigger a collective sentiment wave (like a news event)."""
        if self.active_sentiment_wave:
            return
        
        if np.random.random() < self.sentiment_wave_probability:
            wave = {
                "tick": self.tick,
                "sentiment": np.random.choice(["positive", "negative", "controversial"]),
                "intensity": np.random.uniform(0.2, 0.6),
                "affected_topics": list(np.random.choice(self.n_topics, 
                                          np.random.randint(1, 4), replace=False)),
                "duration": np.random.randint(2, 6),
            }
            self.active_sentiment_wave = wave
    
    def _apply_sentiment_wave(self):
        """Apply the effects of an active sentiment wave to agents."""
        if not self.active_sentiment_wave:
            return
        
        wave = self.active_sentiment_wave
        affected = wave["affected_topics"]
        
        for agent in self.agents:
            agent_exposure = sum(agent.interests[t] for t in affected if t < len(agent.interests))
            
            if agent_exposure > 0.1:
                if wave["sentiment"] == "positive":
                    agent.emotional_state = min(1.0, agent.emotional_state + wave["intensity"] * 0.1)
                elif wave["sentiment"] == "negative":
                    agent.emotional_state = max(-1.0, agent.emotional_state - wave["intensity"] * 0.1)
                elif wave["sentiment"] == "controversial":
                    agent.emotional_state *= (1 + wave["intensity"] * 0.3)
                    agent.emotional_state = np.clip(agent.emotional_state, -1, 1)
    
    def _apply_language_drift(self):
        """Apply language drift to communities."""
        for community_id in self.community_detector.communities:
            self.language_engine.generate_community_slang(community_id, self.tick)
    
    def get_state(self) -> Dict:
        """Get the full simulation state."""
        return {
            "tick": self.tick,
            "n_agents": len(self.agents),
            "n_bots": sum(1 for a in self.agents if a.is_bot),
            "n_posts": len(self.posts),
            "platform": self.platform.to_dict(),
            "meme_evolver": self.meme_evolver.to_dict(),
            "community_detector": self.community_detector.to_dict(),
            "language_engine": self.language_engine.to_dict(),
            "metrics": self.metrics.get_summary(),
            "sentiment_wave": self.active_sentiment_wave,
            "follow_cascade_count": len(self.follow_cascades),
            "neural_net": self.neural_ranker.get_status(),
        }
    
    def reset(self, **kwargs):
        """Reset the simulation with optional new parameters."""
        n_agents = kwargs.get("n_agents", self.n_agents)
        n_topics = kwargs.get("n_topics", self.n_topics)
        echo_chamber = kwargs.get("echo_chamber_strength", self.platform.echo_chamber_strength)
        virality = kwargs.get("virality_sensitivity", self.platform.virality_sensitivity)
        novelty_decay = kwargs.get("novelty_decay", self.platform.novelty_decay)
        mutation_rate = kwargs.get("mutation_rate", self.meme_evolver.mutation_rate)
        
        self.__init__(
            n_agents=n_agents,
            n_topics=n_topics,
            echo_chamber_strength=echo_chamber,
            virality_sensitivity=virality,
            novelty_decay=novelty_decay,
            mutation_rate=mutation_rate,
            max_posts=self.max_posts,
            community_detection_interval=self.community_detection_interval,
            bot_fraction=kwargs.get("bot_fraction", self.bot_fraction),
        )
