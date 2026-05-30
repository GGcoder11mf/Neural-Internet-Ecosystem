"""
Language Drift System for the Fake Internet Ecosystem.

Over time, language evolves within communities:
- Words get compressed for efficiency
- Emojis/symbols replace phrases
- Grammar simplifies inside groups
- New slang emerges independently per community
- Each cluster develops its own dialect
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from .agent import UserAgent
from .community import Community


# Common word compression mappings
COMPRESSION_RULES = {
    "because": "cuz",
    "probably": "prolly",
    "though": "tho",
    "through": "thru",
    "right": "rite",
    "about": "bout",
    "going to": "gonna",
    "want to": "wanna",
    "got to": "gotta",
    "kind of": "kinda",
    "out of": "outta",
    "don't know": "dunno",
    "got you": "gotchu",
    "love you": "ily",
    "thank you": "thx",
    "please": "plz",
    "people": "ppl",
    "someone": "some1",
    "anyone": "any1",
    "everyone": "every1",
    "before": "b4",
    "for real": "fr",
    "to be honest": "tbh",
    "in my opinion": "imo",
    "not gonna lie": "ngl",
    "on god": "ong",
    "i know right": "ikr",
}

# Emoji replacements for common phrases
EMOJI_RULES = {
    "laughing": "😂",
    "fire": "🔥",
    "love": "❤️",
    "skull": "💀",
    "crying": "😭",
    "clown": "🤡",
    "cap": "🧢",
    "eyes": "👀",
    "brain": "🧠",
    "100": "💯",
    "trash": "🗑️",
    "star": "⭐",
    "crown": "👑",
    "lightning": "⚡",
    "check": "✅",
    "cross": "❌",
    "wave": "👋",
    "shrug": "🤷",
    "think": "🤔",
    "cool": "😎",
}

# Community-specific slang that can emerge
EMERGENT_SLANG_POOL = [
    "chad", "based", "sigma", "grindset", "looksmax",
    "mog", "cope", "seethe", "dilate", "touch grass",
    "rent free", "ratio", "w", "l", "mid", "bussin",
    "cap", "no cap", "slaps", "hits different", "vibe check",
    "main character", "npc", "side quest", "glow up",
    "tea", "shade", "sus", "amogus", "poggers",
    "sweat", "tryhard", "sweat", "cracked", "dub",
    "f", "rip", "gg", "ez", "rekt", "pwned",
    "stonks", "doge", "wen", "hodl", "ape",
    "dench", "peng", "bare", "innit", "bruv",
]


class LanguageDriftEngine:
    """
    Manages language evolution across communities.
    
    Tracks word frequencies, applies compression, generates slang,
    and measures language divergence between communities.
    """
    
    def __init__(self, compression_rate: float = 0.1,
                 slang_generation_rate: float = 0.05,
                 emoji_replacement_rate: float = 0.08,
                 grammar_simplification_rate: float = 0.03):
        self.compression_rate = compression_rate
        self.slang_generation_rate = slang_generation_rate
        self.emoji_replacement_rate = emoji_replacement_rate
        self.grammar_simplification_rate = grammar_simplification_rate
        
        # Global vocabulary tracking
        self.global_vocabulary: Counter = Counter()
        self.community_vocabularies: Dict[int, Counter] = defaultdict(Counter)
        
        # Dialect tracking
        self.community_dialects: Dict[int, Dict[str, str]] = {}  # community_id -> {standard -> dialect}
        
        # Emergent slang that has been adopted
        self.adopted_slang: Dict[int, Set[str]] = defaultdict(set)
        
        # Language divergence history
        self.divergence_history: List[float] = []
    
    def process_text(self, text: str, community_id: int, 
                     tick: int) -> str:
        """
        Process text through language drift for a specific community.
        Applies compression, emoji replacement, and community-specific dialect.
        """
        if not text:
            return text
        
        words = text.split()
        processed = []
        
        # Get community dialect
        dialect = self.community_dialects.get(community_id, {})
        
        for word in words:
            new_word = word
            
            # Apply community dialect first
            if word.lower() in dialect:
                new_word = dialect[word.lower()]
            # Apply compression
            elif word.lower() in COMPRESSION_RULES and np.random.random() < self.compression_rate:
                new_word = COMPRESSION_RULES[word.lower()]
            # Apply emoji replacement
            elif word.lower() in EMOJI_RULES and np.random.random() < self.emoji_replacement_rate:
                new_word = EMOJI_RULES[word.lower()]
            
            processed.append(new_word)
        
        # Grammar simplification: occasionally remove articles and prepositions
        if np.random.random() < self.grammar_simplification_rate:
            processed = self._simplify_grammar(processed)
        
        result = " ".join(processed)
        
        # Update vocabulary tracking
        self._update_vocabulary(result, community_id)
        
        return result
    
    def _simplify_grammar(self, words: List[str]) -> List[str]:
        """Remove some grammatical elements for efficiency."""
        articles = {"a", "an", "the", "is", "are", "was", "were", "am"}
        simplified = []
        for word in words:
            if word.lower() not in articles or np.random.random() > 0.5:
                simplified.append(word)
        return simplified if simplified else words
    
    def _update_vocabulary(self, text: str, community_id: int):
        """Track word frequencies globally and per community."""
        words = text.lower().split()
        self.global_vocabulary.update(words)
        self.community_vocabularies[community_id].update(words)
    
    def generate_community_slang(self, community_id: int, tick: int):
        """
        Generate new slang for a community.
        Slang emerges from the community's most-used words being corrupted.
        """
        if community_id not in self.community_vocabularies:
            return
        
        vocab = self.community_vocabularies[community_id]
        if not vocab:
            return
        
        if np.random.random() > self.slang_generation_rate:
            return
        
        # Pick a frequent word to corrupt into slang
        common_words = [w for w, c in vocab.most_common(20) if len(w) > 3]
        if not common_words:
            return
        
        base_word = np.random.choice(common_words)
        
        # Either adopt from pool or corrupt
        if np.random.random() < 0.5 and EMERGENT_SLANG_POOL:
            new_slang = np.random.choice(EMERGENT_SLANG_POOL)
            # Only adopt if not already adopted by another community
            already_adopted = any(new_slang in slang_set 
                                 for cid, slang_set in self.adopted_slang.items() 
                                 if cid != community_id)
            if not already_adopted:
                self.adopted_slang[community_id].add(new_slang)
                if community_id not in self.community_dialects:
                    self.community_dialects[community_id] = {}
                self.community_dialects[community_id][base_word] = new_slang
        else:
            # Corrupt the word
            new_slang = self._corrupt_word(base_word)
            self.adopted_slang[community_id].add(new_slang)
            if community_id not in self.community_dialects:
                self.community_dialects[community_id] = {}
            self.community_dialects[community_id][base_word] = new_slang
    
    def _corrupt_word(self, word: str) -> str:
        """Corrupt a word into slang through various transformations."""
        corruption_type = np.random.choice(["shorten", "repeat", "replace", "combine"])
        
        if corruption_type == "shorten" and len(word) > 3:
            # Take first syllable-like chunk
            return word[:max(2, len(word) // 2)]
        
        elif corruption_type == "repeat":
            # Repeat a letter
            idx = np.random.randint(0, len(word))
            return word[:idx] + word[idx] + word[idx:]
        
        elif corruption_type == "replace":
            # Replace vowels
            vowels = "aeiou"
            result = list(word)
            for i, c in enumerate(result):
                if c in vowels and np.random.random() < 0.3:
                    result[i] = np.random.choice(list(vowels))
            return "".join(result)
        
        elif corruption_type == "combine" and EMERGENT_SLANG_POOL:
            # Combine with a slang term
            slang = np.random.choice(EMERGENT_SLANG_POOL)
            return word[:len(word)//2] + slang[:len(slang)//2]
        
        return word
    
    def compute_language_divergence(self) -> float:
        """
        Compute overall language divergence across communities.
        Higher = communities speak more differently.
        """
        community_ids = list(self.community_vocabularies.keys())
        if len(community_ids) < 2:
            return 0.0
        
        # Compare top words between communities
        divergences = []
        for i in range(len(community_ids)):
            for j in range(i + 1, len(community_ids)):
                ci = community_ids[i]
                cj = community_ids[j]
                
                vocab_i = self.community_vocabularies[ci]
                vocab_j = self.community_vocabularies[cj]
                
                # Jaccard distance of top words
                top_i = set(w for w, _ in vocab_i.most_common(30))
                top_j = set(w for w, _ in vocab_j.most_common(30))
                
                if top_i and top_j:
                    intersection = len(top_i & top_j)
                    union = len(top_i | top_j)
                    jaccard_dist = 1 - (intersection / union if union > 0 else 0)
                    divergences.append(jaccard_dist)
        
        avg_divergence = np.mean(divergences) if divergences else 0.0
        self.divergence_history.append(avg_divergence)
        return avg_divergence
    
    def get_community_dialect_summary(self, community_id: int) -> Dict:
        """Get a summary of a community's dialect."""
        dialect = self.community_dialects.get(community_id, {})
        slang = self.adopted_slang.get(community_id, set())
        vocab = self.community_vocabularies.get(community_id, Counter())
        
        return {
            "community_id": community_id,
            "dialect_mappings": dict(dialect),
            "adopted_slang": list(slang),
            "top_words": vocab.most_common(10),
            "vocabulary_size": len(vocab),
        }
    
    def to_dict(self) -> dict:
        return {
            "global_vocab_size": len(self.global_vocabulary),
            "communities_with_dialects": len(self.community_dialects),
            "total_slang_adopted": sum(len(s) for s in self.adopted_slang.values()),
            "language_divergence": self.divergence_history[-1] if self.divergence_history else 0.0,
        }
