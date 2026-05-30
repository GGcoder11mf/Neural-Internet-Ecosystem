"""
Meme Evolution System for the Fake Internet Ecosystem.

When agents repost content, they can mutate it: shorten, shift meaning,
add slang, or merge with another meme. Over time, memes evolve like
genetic drift, and "successful memes" survive attention pressure.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from .content import Meme, Post, ContentType


class MemeEvolver:
    """
    Manages meme evolution across the ecosystem.
    
    Tracks meme lineages, mutation rates, and survival patterns.
    Memes evolve like genetic drift under attention pressure.
    """
    
    def __init__(self, mutation_rate: float = 0.3, 
                 merge_probability: float = 0.1,
                 selection_pressure: float = 0.5):
        self.mutation_rate = mutation_rate
        self.merge_probability = merge_probability
        self.selection_pressure = selection_pressure
        
        # Meme lineage tracking
        self.meme_families: Dict[str, List[str]] = {}  # root_id -> [descendant_ids]
        self.meme_lookup: Dict[str, Meme] = {}
        
        # Evolution statistics
        self.total_mutations = 0
        self.total_merges = 0
        self.extinct_memes = 0
        self.dominant_memes: List[str] = []
        
        # Mutation type frequencies
        self.mutation_type_counts: Dict[str, int] = {
            "shorten": 0, "shift": 0, "slang": 0, "merge": 0
        }
    
    def should_mutate(self, agent_personality: str = "") -> bool:
        """Determine if a repost should include a mutation."""
        base_rate = self.mutation_rate
        
        # Meme lords mutate much more
        personality_boost = {
            "meme_lord": 0.5,
            "toxic": 0.15,
            "curious": 0.1,
            "viral_chaser": 0.05,
            "influencer": 0.08,
        }
        base_rate += personality_boost.get(agent_personality, 0.0)
        
        return np.random.random() < base_rate
    
    def should_merge(self, available_memes: List[Meme]) -> bool:
        """Determine if two memes should be merged."""
        if len(available_memes) < 2:
            return False
        return np.random.random() < self.merge_probability
    
    def evolve_meme(self, parent_meme: Meme, mutator_id: str,
                    current_tick: int, 
                    available_memes: List[Meme] = None) -> Meme:
        """
        Evolve a meme through mutation or merge.
        
        Returns a new child meme.
        """
        # Check for merge
        if available_memes and self.should_merge(available_memes):
            other = self._pick_merge_partner(parent_meme, available_memes)
            if other:
                child = parent_meme.merge_with(other, mutator_id)
                child.tick_created = current_tick
                self.total_merges += 1
                self.mutation_type_counts["merge"] += 1
                self._register_meme(child, parent_meme.id)
                return child
        
        # Regular mutation
        mutation_type = self._pick_mutation_type()
        child = parent_meme.mutate(mutator_id, mutation_type)
        child.tick_created = current_tick
        
        self.total_mutations += 1
        self.mutation_type_counts[mutation_type] += 1
        self._register_meme(child, parent_meme.id)
        
        return child
    
    def _pick_mutation_type(self) -> str:
        """Pick a mutation type based on weighted probabilities."""
        types = ["shorten", "shift", "slang"]
        weights = [0.3, 0.35, 0.35]
        return np.random.choice(types, p=weights)
    
    def _pick_merge_partner(self, meme: Meme, candidates: List[Meme]) -> Optional[Meme]:
        """Pick a suitable meme to merge with."""
        # Prefer memes with similar but not identical topics
        best_partner = None
        best_score = -1
        
        for candidate in candidates:
            if candidate.id == meme.id:
                continue
            
            similarity = np.dot(meme.topic_vector, candidate.topic_vector)
            # Prefer moderate similarity (not too similar, not too different)
            merge_score = 1.0 - abs(similarity - 0.5)
            
            if merge_score > best_score and np.random.random() < 0.3:
                best_score = merge_score
                best_partner = candidate
        
        # Fallback: random
        if best_partner is None and candidates:
            others = [m for m in candidates if m.id != meme.id]
            if others:
                best_partner = np.random.choice(others)
        
        return best_partner
    
    def _register_meme(self, meme: Meme, parent_id: str):
        """Register a new meme in the lineage tracker."""
        self.meme_lookup[meme.id] = meme
        
        # Find root
        root_id = parent_id
        if parent_id in self.meme_lookup:
            parent = self.meme_lookup[parent_id]
            # Walk up to root
            current = parent
            while current.parent_meme_id and current.parent_meme_id in self.meme_lookup:
                current = self.meme_lookup[current.parent_meme_id]
            root_id = current.id
        
        if root_id not in self.meme_families:
            self.meme_families[root_id] = [root_id]
        if meme.id not in self.meme_families[root_id]:
            self.meme_families[root_id].append(meme.id)
    
    def apply_selection_pressure(self, memes: List[Meme], attention_scores: np.ndarray):
        """
        Apply selection pressure: memes with low attention may go extinct.
        """
        if not memes or len(attention_scores) == 0:
            return
        
        # Compute survival threshold
        median_attention = np.median(attention_scores)
        threshold = median_attention * (1 - self.selection_pressure)
        
        for i, meme in enumerate(memes):
            if attention_scores[i] < threshold:
                # Mark as potentially extinct
                if meme.id in self.meme_lookup:
                    self.extinct_memes += 1
    
    def update_dominant_memes(self, memes: List[Meme], attention_scores: np.ndarray,
                               top_k: int = 5):
        """Track which memes dominate the ecosystem."""
        if not memes:
            return
        
        meme_attention = [(meme.id, attention_scores[i]) for i, meme in enumerate(memes)]
        meme_attention.sort(key=lambda x: x[1], reverse=True)
        self.dominant_memes = [m[0] for m in meme_attention[:top_k]]
    
    def compute_meme_diversity(self, memes: List[Meme]) -> float:
        """
        Compute meme diversity using average pairwise distance.
        Higher = more diverse ecosystem.
        """
        if len(memes) < 2:
            return 0.0
        
        vectors = np.array([m.topic_vector for m in memes])
        n = len(vectors)
        
        # Sample for performance if too many
        if n > 100:
            indices = np.random.choice(n, 100, replace=False)
            vectors = vectors[indices]
            n = 100
        
        # Compute pairwise distances
        similarity_matrix = vectors @ vectors.T
        diversity = 1.0 - np.mean(similarity_matrix)
        return max(0, diversity)
    
    def get_meme_lifespan_stats(self, current_tick: int) -> Dict:
        """Compute statistics about meme lifespans."""
        if not self.meme_lookup:
            return {"mean_lifespan": 0, "max_lifespan": 0, "median_lifespan": 0}
        
        lifespans = []
        for meme in self.meme_lookup.values():
            lifespan = current_tick - meme.tick_created
            lifespans.append(lifespan)
        
        if not lifespans:
            return {"mean_lifespan": 0, "max_lifespan": 0, "median_lifespan": 0}
        
        return {
            "mean_lifespan": np.mean(lifespans),
            "max_lifespan": np.max(lifespans),
            "median_lifespan": np.median(lifespan for lifespan in [lifespans]),
        }
    
    def get_family_tree(self, root_id: str) -> Dict:
        """Get the family tree of a meme lineage."""
        if root_id not in self.meme_families:
            return {}
        
        family = self.meme_families[root_id]
        tree = {"root": root_id, "descendants": [], "size": len(family)}
        
        for meme_id in family:
            if meme_id in self.meme_lookup:
                meme = self.meme_lookup[meme_id]
                tree["descendants"].append({
                    "id": meme.id,
                    "text": meme.text[:50],
                    "generation": meme.generation,
                    "mutations": len(meme.mutation_history),
                })
        
        return tree
    
    def to_dict(self) -> dict:
        return {
            "total_mutations": self.total_mutations,
            "total_merges": self.total_merges,
            "extinct_memes": self.extinct_memes,
            "dominant_memes": self.dominant_memes,
            "mutation_type_counts": self.mutation_type_counts,
            "num_families": len(self.meme_families),
            "total_tracked_memes": len(self.meme_lookup),
        }
