"""
Content system for the Fake Internet Ecosystem.

Contains Post, Meme, and Idea classes representing content objects.
Richer post generation with personality-driven templates, hashtags,
emoji usage, information distortion (telephone game), and conversation threading.

Realism improvements:
- 100+ post templates across 10 personality types
- Hashtags naturally generated from topic vectors
- Emoji usage varies by personality
- Information distortion when content is reposted/relayed
- Quote tweet support (repost with commentary)
- Reply threading
- Controversy scoring
- Engagement bait detection
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import uuid
import re


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    IDEA = "idea"
    MEME = "meme"
    QUOTE = "quote"      # Quote tweet / repost with comment
    THREAD = "thread"    # Part of a thread


class Post:
    """A content object in the ecosystem."""
    
    def __init__(self, author_id: str, content_type: ContentType = ContentType.TEXT,
                 text: str = "", topic_vector: np.ndarray = None,
                 n_topics: int = 20, tick_created: int = 0,
                 parent_id: str = None, hashtags: List[str] = None,
                 quote_text: str = "", is_bot_post: bool = False):
        self.id = str(uuid.uuid4())[:10]
        self.author_id = author_id
        self.content_type = content_type
        self.text = text
        self.tick_created = tick_created
        self.parent_id = parent_id  # For replies/remixes
        self.hashtags = hashtags or []
        self.quote_text = quote_text  # For quote tweets
        self.is_bot_post = is_bot_post
        
        # Topic/meaning vector
        if topic_vector is not None:
            self.topic_vector = topic_vector.copy()
        else:
            self.topic_vector = np.random.dirichlet(np.ones(n_topics) * 0.5)
        
        # Normalize topic vector
        norm = np.linalg.norm(self.topic_vector)
        if norm > 0:
            self.topic_vector = self.topic_vector / norm
        
        # Novelty score (1.0 = completely novel, decays over time)
        self.novelty_score = 1.0
        
        # Engagement metrics
        self.views = 0
        self.likes = 0
        self.reposts = 0
        self.replies = 0
        self.ignores = 0
        self.quotes = 0  # Quote tweets
        
        # Mutation history (for meme tracking)
        self.mutation_history: List[Dict] = []
        self.generation = 0  # 0 = original, 1+ = remix generation
        
        # Attention tracking
        self.attention_score = 0.0
        self.peak_attention = 0.0
        self.attention_history: List[float] = []
        
        # Controversy score: high when replies >> likes or when engagement is polarized
        self.controversy_score = 0.0
        
        # Distortion tracking: how much has this content been distorted from original?
        self.distortion_level = 0.0  # 0 = original, higher = more distorted
        self.original_text = text  # Track what it originally said
        
        # Embedding hash (for deduplication)
        self.content_hash = hash(text) if text else hash(self.id)
        
        # Thread tracking
        self.thread_id: Optional[str] = None  # Which thread this belongs to
        self.thread_position = 0  # Position in thread (1 = first tweet)
        
        # Image path (for meme/post images)
        self.image_path: Optional[str] = None
    
    @property
    def engagement_rate(self) -> float:
        """Compute engagement rate."""
        if self.views == 0:
            return 0.0
        return (self.likes + self.reposts + self.replies + self.quotes) / max(self.views, 1)
    
    @property
    def total_engagement(self) -> int:
        return self.likes + self.reposts + self.replies + self.quotes
    
    @property
    def ratio_score(self) -> float:
        """
        Compute 'ratio' — when replies/quotes >> likes, content is controversial/bad.
        A high ratio_score means people are arguing with or mocking this post.
        """
        if self.likes == 0:
            return float(self.replies + self.quotes)
        return (self.replies + self.quotes) / self.likes
    
    def age_at(self, current_tick: int) -> int:
        return current_tick - self.tick_created
    
    def compute_novelty(self, current_tick: int, decay: float = 0.95) -> float:
        """Decay novelty over time."""
        age = self.age_at(current_tick)
        self.novelty_score = decay ** age
        return self.novelty_score
    
    def compute_controversy(self):
        """
        Compute controversy score based on engagement patterns.
        High controversy = lots of replies/quotes relative to likes.
        Real social media: controversial posts get ratio'd.
        """
        total_engagement = self.likes + self.reposts + self.replies + self.quotes
        if total_engagement < 3:
            self.controversy_score = 0.0
            return self.controversy_score
        
        # High reply-to-like ratio = controversy
        reply_ratio = self.ratio_score
        # Also: high engagement but low likes = controversial
        engagement_without_likes = self.reposts + self.replies + self.quotes
        
        # Normalize: ratio > 1 is getting controversial, > 3 is very controversial
        self.controversy_score = min(1.0, reply_ratio / 5.0)
        
        # Boost if engagement is high but likes are disproportionately low
        if engagement_without_likes > self.likes * 2 and engagement_without_likes > 5:
            self.controversy_score = min(1.0, self.controversy_score + 0.2)
        
        return self.controversy_score
    
    def record_attention(self, attention: float):
        """Record attention score for this tick."""
        self.attention_score = attention
        self.attention_history.append(attention)
        self.peak_attention = max(self.peak_attention, attention)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "content_type": self.content_type.value,
            "text": self.text[:200],
            "tick_created": self.tick_created,
            "views": self.views,
            "likes": self.likes,
            "reposts": self.reposts,
            "replies": self.replies,
            "quotes": self.quotes,
            "engagement_rate": round(self.engagement_rate, 3),
            "novelty_score": round(self.novelty_score, 3),
            "attention_score": round(self.attention_score, 3),
            "peak_attention": round(self.peak_attention, 3),
            "generation": self.generation,
            "mutation_count": len(self.mutation_history),
            "controversy_score": round(self.controversy_score, 3),
            "distortion_level": round(self.distortion_level, 3),
            "hashtags": self.hashtags,
            "is_bot_post": self.is_bot_post,
            "ratio_score": round(self.ratio_score, 2),
        }


class Meme(Post):
    """A meme - a special type of post that can mutate and evolve."""
    
    def __init__(self, author_id: str, text: str = "", 
                 topic_vector: np.ndarray = None,
                 n_topics: int = 20, tick_created: int = 0,
                 parent_meme_id: str = None):
        super().__init__(
            author_id=author_id,
            content_type=ContentType.MEME,
            text=text,
            topic_vector=topic_vector,
            n_topics=n_topics,
            tick_created=tick_created,
            parent_id=parent_meme_id
        )
        self.parent_meme_id = parent_meme_id
        self.mutation_count = 0
        self.children_meme_ids: List[str] = []
        self.dominance_score = 0.0
    
    def mutate(self, mutator_id: str, mutation_type: str = None) -> 'Meme':
        """
        Create a mutated child meme.
        
        Mutation types:
        - shorten: Remove words from text
        - shift: Slightly change topic vector
        - slang: Add slang/corruption to text
        - distort: Information distortion (telephone game)
        - merge: Combine with another meme (handled separately)
        """
        if mutation_type is None:
            mutation_type = np.random.choice(
                ["shorten", "shift", "slang", "distort", "slang"],
                p=[0.2, 0.2, 0.25, 0.2, 0.15]
            )
        
        new_text = self.text
        new_topic = self.topic_vector.copy()
        
        if mutation_type == "shorten":
            words = new_text.split()
            if len(words) > 3:
                n_remove = min(np.random.randint(1, 4), len(words) - 2)
                indices = np.random.choice(len(words), n_remove, replace=False)
                words = [w for i, w in enumerate(words) if i not in indices]
                new_text = " ".join(words)
        
        elif mutation_type == "shift":
            noise = np.random.normal(0, 0.05, new_topic.shape)
            new_topic = new_topic + noise
            norm = np.linalg.norm(new_topic)
            if norm > 0:
                new_topic = new_topic / norm
        
        elif mutation_type == "slang":
            slang_additions = [
                "lol", "ngl", "fr", "no cap", "lowkey", "highkey", 
                "bet", "based", "cringe", "slay", "vibe", "pog",
                "fam", "bruh", "sheesh", "wild", "fire", "mid",
                "w", "l", "ratio", "rent free", "touch grass",
                "gg", "ez", "f", "rip", "tbh", "imo", "iykyk",
                "aint no way", "deadass", "bussin", "no shot",
                "its giving", "era", "slaps", "hits different",
            ]
            addition = np.random.choice(slang_additions)
            if np.random.random() < 0.5:
                new_text = f"{addition} {new_text}"
            else:
                new_text = f"{new_text} {addition}"
        
        elif mutation_type == "distort":
            # TELEPHONE GAME: when content is relayed, it gets distorted
            # People misremember, exaggerate, simplify, or twist meaning
            distort_type = np.random.choice([
                "exaggerate", "simplify", "twist", "misquote", "condense"
            ])
            
            if distort_type == "exaggerate":
                # Add intensifiers
                intensifiers = ["literally", "actually", "absolutely", "completely",
                                "insanely", "unbelievably", "extremely"]
                words = new_text.split()
                if len(words) > 2:
                    insert_pos = np.random.randint(1, len(words))
                    words.insert(insert_pos, np.random.choice(intensifiers))
                    new_text = " ".join(words)
            
            elif distort_type == "simplify":
                # Remove qualifiers and nuance
                nuance_words = {"maybe", "perhaps", "might", "could", "sometimes",
                                "often", "usually", "somewhat", "slightly", "fairly",
                                "rather", "quite", "apparently", "seemingly", "possibly"}
                words = new_text.split()
                words = [w for w in words if w.lower() not in nuance_words]
                if words:
                    new_text = " ".join(words)
            
            elif distort_type == "twist":
                # Flip a negative/positive word
                flips = {
                    "good": "terrible", "bad": "amazing", "love": "hate",
                    "great": "awful", "best": "worst", "right": "wrong",
                    "smart": "stupid", "beautiful": "ugly", "important": "irrelevant",
                    "true": "false", "real": "fake", "helpful": "harmful",
                    "safe": "dangerous", "easy": "impossible",
                }
                words = new_text.split()
                for i, w in enumerate(words):
                    if w.lower() in flips and np.random.random() < 0.5:
                        words[i] = flips[w.lower()]
                        break
                new_text = " ".join(words)
            
            elif distort_type == "misquote":
                # Replace a key word with something else
                words = new_text.split()
                if len(words) > 3:
                    replace_idx = np.random.randint(0, len(words))
                    # Replace with a related but different word
                    replacements = ["apparently", "supposedly", "rumored", "claimed",
                                    "sources say", "reports indicate"]
                    words[replace_idx] = np.random.choice(replacements)
                    new_text = " ".join(words)
            
            elif distort_type == "condense":
                # Take only the first part (lose context)
                words = new_text.split()
                if len(words) > 4:
                    keep = max(2, len(words) // 2)
                    new_text = " ".join(words[:keep]) + "..."
        
        child = Meme(
            author_id=mutator_id,
            text=new_text,
            topic_vector=new_topic,
            n_topics=len(new_topic),
            tick_created=self.tick_created,
            parent_meme_id=self.id
        )
        child.generation = self.generation + 1
        child.mutation_count = self.mutation_count + 1
        child.mutation_history = self.mutation_history + [{
            "type": mutation_type,
            "parent_id": self.id,
            "generation": child.generation,
        }]
        child.distortion_level = self.distortion_level + (0.2 if mutation_type == "distort" else 0.05)
        child.original_text = self.original_text
        
        self.children_meme_ids.append(child.id)
        
        return child
    
    def merge_with(self, other: 'Meme', merger_id: str) -> 'Meme':
        """Merge two memes into a new combined meme."""
        words_a = self.text.split()
        words_b = other.text.split()
        
        n_a = max(1, len(words_a) // 2)
        n_b = max(1, len(words_b) // 2)
        
        combined = " ".join(words_a[:n_a]) + " " + " ".join(words_b[:n_b])
        
        new_topic = (self.topic_vector + other.topic_vector) / 2
        norm = np.linalg.norm(new_topic)
        if norm > 0:
            new_topic = new_topic / norm
        
        child = Meme(
            author_id=merger_id,
            text=combined,
            topic_vector=new_topic,
            n_topics=len(new_topic),
            tick_created=self.tick_created,
            parent_meme_id=self.id
        )
        child.generation = max(self.generation, other.generation) + 1
        child.mutation_count = self.mutation_count + other.mutation_count + 1
        child.mutation_history = self.mutation_history + other.mutation_history + [{
            "type": "merge",
            "parent_ids": [self.id, other.id],
            "generation": child.generation,
        }]
        child.distortion_level = max(self.distortion_level, other.distortion_level) + 0.15
        child.original_text = self.original_text
        
        return child


class Idea(Post):
    """An idea - compressed meaning vector content."""
    
    def __init__(self, author_id: str, meaning_vector: np.ndarray = None,
                 n_topics: int = 20, tick_created: int = 0,
                 label: str = ""):
        super().__init__(
            author_id=author_id,
            content_type=ContentType.IDEA,
            text=label,
            topic_vector=meaning_vector,
            n_topics=n_topics,
            tick_created=tick_created
        )
        self.meaning_vector = self.topic_vector.copy()
        self.label = label
        self.compression_level = 0
    
    def compress(self) -> 'Idea':
        """Compress the idea - reduce dimensionality information while preserving core meaning."""
        compressed = self.meaning_vector.copy()
        threshold = np.percentile(np.abs(compressed), 30)
        compressed[np.abs(compressed) < threshold] = 0
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed = compressed / norm
        
        child = Idea(
            author_id=self.author_id,
            meaning_vector=compressed,
            n_topics=len(compressed),
            tick_created=self.tick_created,
            label=self.label
        )
        child.compression_level = self.compression_level + 1
        return child


# ========================================
# CONTENT GENERATION — Realistic Templates
# ========================================

TOPIC_LABELS = [
    "tech", "politics", "gaming", "food", "travel",
    "science", "art", "music", "sports", "fashion",
    "crypto", "memes", "philosophy", "nature", "history",
    "health", "education", "entertainment", "business", "culture"
]

# Per-personality post templates — much more varied and realistic
PERSONALITY_TEMPLATES = {
    "curious": [
        "Has anyone else noticed {topic}?",
        "Genuine question: what's the deal with {topic}?",
        "I've been researching {topic} and wow, there's more to it than I thought",
        "Can someone explain {topic} to me like I'm 5?",
        "What's the most surprising thing about {topic} that most people don't know?",
        "Just went down a rabbit hole about {topic} and my mind is blown",
        "Is it just me or has {topic} gotten way more complicated lately?",
        "I keep finding new layers to {topic} every time I look into it",
        "What's a controversial take on {topic} that actually makes sense?",
        "TIL that {topic} has a whole hidden world most people miss",
        "Can we talk about the intersection of {topic} and {topic2}?",
        "What would happen if we completely rethought how we approach {topic}?",
        "I just realized {topic} connects to basically everything",
        "The more I learn about {topic}, the less I feel like I understand",
    ],
    "toxic": [
        "Bro {topic} is literally the worst thing ever",
        "Anyone who likes {topic} has zero taste change my mind",
        "{topic} is trash and so are the people who support it",
        "Imagine caring about {topic} in {year} lmaooo",
        "Not people still talking about {topic} 💀💀💀",
        "The {topic} community is so cringe I can't even",
        "Why is {topic} so overrated? The hype is embarrassing",
        "I'm sick of seeing {topic} everywhere, touch grass people",
        "{topic} stans are the worst part of the internet",
        "If you think {topic} is good you're part of the problem",
        "L take on {topic}, try again",
        "Ratio + {topic} is mid + didn't ask",
        "The way y'all glaze {topic} is insane",
    ],
    "viral_chaser": [
        "THIS IS HUGE: {topic} just broke the internet",
        "POV: you just discovered {topic} and you're obsessed",
        "The way {topic} is taking over rn is insane",
        "Everyone needs to see this {topic} update right now",
        "I was today years old when I found out about {topic}",
        "Not {topic} breaking the internet again",
        "THIS {topic} TAKE WILL CHANGE YOUR LIFE",
        "I cannot stress enough how important this {topic} moment is",
        "{topic} is having a moment and I'm here for it",
        "Tell me you're trending without telling me you're trending: {topic}",
        "If you're not talking about {topic} rn what ARE you doing",
        "The internet is not ready for this {topic} take",
        "Breaking: {topic} just went viral and here's why",
    ],
    "niche_thinker": [
        "Interesting pattern emerging in {topic}: the underlying dynamics suggest a shift",
        "The relationship between {topic} and structural incentives is underappreciated",
        "A thread on why {topic} discourse misses the fundamental point",
        "What nobody mentions about {topic}: the second-order effects",
        "The real question about {topic} isn't what you think it is",
        "Been mapping the {topic} ecosystem and found some fascinating connections",
        "The meta-problem with {topic} analysis is the framing itself",
        "Unpopular but: {topic} requires looking at incentives, not intentions",
        "The {topic} debate is fundamentally miscategorized and here's my reasoning",
        "Why {topic} analogies fail: they ignore the structural differences",
        "A framework for understanding {topic} that I've been developing",
        "The hidden variable in {topic} discourse is incentive alignment",
        "Most {topic} analysis suffers from selection bias",
    ],
    "lurker": [
        "just watching the {topic} drama unfold from the sidelines",
        "lurking and learning about {topic}",
        "not me reading every single {topic} thread at 2am",
        "the {topic} discourse is wild but I'm just here for the show",
        "quietly taking notes on {topic}",
        "can't stop reading about {topic} even though I have nothing to add",
        "the {topic} timeline is doing things to my brain",
        "hours deep into {topic} threads and I have no regrets",
        "I've read 47 posts about {topic} today and I'm not stopping",
        "just here, existing, watching {topic} unfold",
    ],
    "influencer": [
        "Y'all need to hear this about {topic}",
        "Okay but can we actually discuss {topic} for a second?",
        "I've been wanting to talk about {topic} and today's the day",
        "Hot take incoming: {topic} is about to change everything",
        "The conversation around {topic} needs to shift, and here's why",
        "I don't think people realize how big {topic} is about to get",
        "Let me share something about {topic} that changed my perspective",
        "My thoughts on {topic} (a thread):",
        "Real talk about {topic} — no one is saying this but they should",
        "If you care about {topic}, this is the post you need to read",
        "Breaking my silence on {topic}",
        "The truth about {topic} that nobody wants to admit",
    ],
    "meme_lord": [
        "ngl {topic} got me acting up",
        "me explaining {topic} to my last brain cell",
        "the way {topic} has me in a chokehold",
        "{topic} but make it ✨aesthetic✨",
        "ain't no way {topic} just happened like that",
        "me when {topic}: 🤡",
        "the {topic} agenda is real and I'm here for it",
        "{topic} vibes are immaculate rn",
        "no because {topic} literally activated my fight or flight",
        "it's giving {topic} and I'm not mad about it",
        "me trying to process {topic}:",
        "the way {topic} just shifted my entire worldview",
        "{topic} took personally and I respect that",
    ],
    "echo_seeker": [
        "Finally someone said what we've all been thinking about {topic}",
        "This is exactly why I've been saying {topic} is important",
        "Who else feels this way about {topic}? I know I'm not alone",
        "Validation: {topic} really is as {adj} as I thought",
        "The {topic} community gets it. Everyone else is missing the point",
        "Glad to see people finally coming around on {topic}",
        "See, THIS is what I mean about {topic}",
        "I've been saying this about {topic} for so long",
        "Our take on {topic} is the correct one and I will die on this hill",
        "Finally a good take on {topic}",
        "The right people understand {topic}, the rest are just loud",
    ],
    "doom_scroller": [
        "can't stop reading about {topic} even though it's destroying my mental health",
        "why am I still awake reading about {topic} at 3am",
        "every time I think {topic} can't get worse it does",
        "the {topic} timeline is a car crash I can't look away from",
        "just spent 2 hours doom scrolling {topic} and I feel terrible",
        "is it just me or has {topic} gotten increasingly bleak",
        "I know I should log off but {topic} keeps pulling me back",
        "the state of {topic} is genuinely concerning and no one cares",
        "I hate that I can't stop thinking about {topic}",
        "scrolling through {topic} threads like it's my job",
        "the {topic} discourse is bad for my health but here I am",
    ],
    "hot_taker": [
        "Unpopular opinion: {topic} is actually {contrary_adj}",
        "HOT TAKE: {topic} is {contrary_adj} and I will NOT be elaborating",
        "I said what I said about {topic}",
        "Come fight me on this: {topic} is overrated and here's why",
        "The most {contrary_adj} take on {topic} you'll read today",
        "{topic} is {contrary_adj} and y'all just aren't ready for that conversation",
        "I will NOT be taking criticism on my {topic} take at this time",
        "Controversial but true: {topic} is {contrary_adj}",
        "This {topic} take will probably get me cancelled but I don't care",
        "Everyone's wrong about {topic} and I can prove it",
        "The brave take on {topic} that nobody has the guts to say",
        "Sorry not sorry but {topic} is absolutely {contrary_adj}",
        "I'm right about {topic} and history will prove me right",
    ],
    # Bot templates (for spammer bots)
    "bot_spam": [
        "AMAZING {topic} opportunity! Don't miss out! Click here for more",
        "BREAKING: {topic} just hit a new milestone! Act now!",
        "Exclusive {topic} analysis - free access for limited time!",
        "You won't BELIEVE what just happened with {topic}",
        "{topic} is about to EXPLODE. Get in while you can!",
        "URGENT: {topic} update you need to see right now",
        "The secret about {topic} they don't want you to know",
        "FREE {topic} guide - limited spots available!",
    ],
}

# Emoji pools per personality
PERSONALITY_EMOJI = {
    "curious": ["🤔", "💡", "🔍", "🧐", "👀", "🤯"],
    "toxic": ["💀", "🤡", "🗑️", "👎", "😤", "🙄"],
    "viral_chaser": ["🔥", "😱", "⚡", "💥", "🚀", "🌟"],
    "niche_thinker": ["📊", "🧠", "📐", "🔬", "📝", "🎯"],
    "lurker": ["👀", "👻", "🤫", "😅", "🍿", "🫣"],
    "influencer": ["✨", "💪", "👑", "💫", "🎤", "🫶"],
    "meme_lord": ["😂", "😭", "💀", "🤣", "✨", "💀"],
    "echo_seeker": ["💯", "🙌", "✊", "🤝", "💪", "👏"],
    "doom_scroller": ["😰", "😭", "🫠", "😵", "😔", "😮‍💨"],
    "hot_taker": ["☕", "🔥", "💪", "⚡", "🎯", "👊"],
    "bot_spam": ["🚀", "💰", "🔥", "⚡", "📈", "💎"],
}

# Adjectives for personality-driven content
ADJECTIVES = {
    "positive": ["amazing", "incredible", "beautiful", "powerful", "fascinating",
                 "brilliant", "revolutionary", "stunning", "essential", "groundbreaking"],
    "negative": ["terrible", "disastrous", "alarming", "disgusting", "horrifying",
                 "devastating", "pathetic", "shameful", "dangerous", "toxic"],
    "neutral": ["interesting", "notable", "significant", "complex", "evolving",
                "nuanced", "multifaceted", "unprecedented", "curious", "transformative"],
    "contrary": ["overrated", "underrated", "backwards", "misunderstood", "fake",
                 "rigged", "a scam", "the opposite of what you think", "not what it seems",
                 "mid", "overhyped", "underappreciated"],
}


def generate_hashtags(topic_vector: np.ndarray, n_topics: int, personality: str = "",
                      hashtag_usage: float = 0.3) -> List[str]:
    """Generate hashtags based on topic vector and personality."""
    hashtags = []
    
    # Determine number of hashtags (0-3)
    n_hashtags = np.random.choice([0, 1, 1, 2, 2, 3], p=[1-hashtag_usage, hashtag_usage*0.3, hashtag_usage*0.3, hashtag_usage*0.2, hashtag_usage*0.1, hashtag_usage*0.1])
    
    if n_hashtags == 0:
        return hashtags
    
    # Pick top topics
    top_indices = np.argsort(topic_vector)[-min(3, n_topics):][::-1]
    
    for i in range(min(n_hashtags, len(top_indices))):
        idx = top_indices[i]
        base_label = TOPIC_LABELS[idx % len(TOPIC_LABELS)]
        
        # 70% use base label, 30% add variation
        if np.random.random() < 0.7:
            hashtags.append(f"#{base_label}")
        else:
            variations = [
                f"#{base_label}tok", f"#{base_label}life", f"#{base_label}check",
                f"#{base_label}daily", f"#{base_label}community", f"#{base_label}update",
            ]
            hashtags.append(np.random.choice(variations))
    
    return hashtags


def add_emojis(text: str, personality: str, emoji_usage: float = 0.2) -> str:
    """Add emojis to text based on personality and usage rate."""
    if np.random.random() > emoji_usage:
        return text
    
    emoji_pool = PERSONALITY_EMOJI.get(personality, ["✨"])
    n_emoji = np.random.choice([1, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
    
    emojis = " ".join(np.random.choice(emoji_pool, n_emoji))
    
    # Place emojis: end of text most common, sometimes beginning
    if np.random.random() < 0.85:
        return f"{text} {emojis}"
    else:
        return f"{emojis} {text}"


def distort_repost_text(original_text: str, distortion_probability: float = 0.15) -> Tuple[str, float]:
    """
    Apply information distortion when someone reposts/relays content.
    Like the telephone game — each relay slightly changes the message.
    
    Returns (distorted_text, distortion_amount)
    """
    if np.random.random() > distortion_probability:
        return original_text, 0.0
    
    words = original_text.split()
    if len(words) < 3:
        return original_text, 0.0
    
    distortion_type = np.random.choice([
        "drop_word", "add_filler", "swap_similar", "truncate", "add_hedge"
    ])
    
    distortion_amount = 0.0
    
    if distortion_type == "drop_word" and len(words) > 4:
        # Drop a random word (information loss)
        idx = np.random.randint(1, len(words) - 1)
        words.pop(idx)
        distortion_amount = 0.1
    
    elif distortion_type == "add_filler":
        # Add filler words (dilution)
        fillers = ["like", "basically", "honestly", "literally", "apparently",
                   "supposedly", "kind of", "sort of"]
        idx = np.random.randint(0, len(words))
        words.insert(idx, np.random.choice(fillers))
        distortion_amount = 0.08
    
    elif distortion_type == "swap_similar":
        # Swap a word with a similar but different one
        swaps = {
            "some": "most", "many": "all", "could": "will", "might": "definitely",
            "suggests": "proves", "indicates": "confirms", "possible": "certain",
            "often": "always", "rarely": "never", "sometimes": "usually",
        }
        for i, w in enumerate(words):
            if w.lower() in swaps and np.random.random() < 0.3:
                words[i] = swaps[w.lower()]
                distortion_amount = 0.2
                break
    
    elif distortion_type == "truncate":
        # Truncate the text (lose context at the end)
        if len(words) > 5:
            keep = np.random.randint(max(3, len(words) // 2), len(words))
            words = words[:keep]
            distortion_amount = 0.15
    
    elif distortion_type == "add_hedge":
        # Add hedging language that wasn't there (softens claims)
        hedges = ["apparently", "supposedly", "reportedly", "some say",
                  "rumor has it", "word on the street"]
        words.insert(0, np.random.choice(hedges) + ",")
        distortion_amount = 0.12
    
    return " ".join(words), distortion_amount


def generate_post_text(topic_vector: np.ndarray, personality_name: str = "",
                       emoji_usage: float = 0.2, hashtag_usage: float = 0.3,
                       is_bot: bool = False, n_topics: int = 20) -> Tuple[str, List[str]]:
    """
    Generate a post text based on topic vector and personality.
    Returns (text, hashtags).
    
    Much more realistic than the original — personality-driven templates,
    emoji, hashtags, and varied writing styles.
    """
    # Pick the dominant topics
    dominant_idx = np.argmax(topic_vector)
    secondary_idx = np.argsort(topic_vector)[-2] if len(topic_vector) > 1 else dominant_idx
    
    topic_label = TOPIC_LABELS[dominant_idx % len(TOPIC_LABELS)]
    topic2_label = TOPIC_LABELS[secondary_idx % len(TOPIC_LABELS)]
    
    # Select template pool based on personality
    if is_bot:
        pool_key = "bot_spam"
    else:
        pool_key = personality_name if personality_name in PERSONALITY_TEMPLATES else "curious"
    
    templates = PERSONALITY_TEMPLATES[pool_key]
    template = np.random.choice(templates)
    
    # Fill template
    adj = ""
    contrary_adj = ""
    if "{adj}" in template:
        adj = np.random.choice(ADJECTIVES["positive"] + ADJECTIVES["neutral"])
    if "{contrary_adj}" in template:
        contrary_adj = np.random.choice(ADJECTIVES["contrary"])
    
    text = template.format(
        topic=topic_label, 
        topic2=topic2_label,
        adj=adj,
        contrary_adj=contrary_adj,
        year=2024 + np.random.randint(0, 3)
    )
    
    # Add emojis based on personality
    text = add_emojis(text, pool_key, emoji_usage)
    
    # Generate hashtags
    hashtags = generate_hashtags(topic_vector, n_topics, personality_name, hashtag_usage)
    
    return text, hashtags
