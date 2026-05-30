"""
Community Formation System for the Fake Internet Ecosystem.

Agents naturally cluster based on shared interests, repeated interactions,
and algorithm reinforcement. This creates subreddits-like groups,
toxic echo chambers, and niche micro-communities.
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from .agent import UserAgent


class Community:
    """A community of agents with shared interests and interactions."""
    
    def __init__(self, community_id: int, name: str = ""):
        self.id = community_id
        self.name = name or f"community_{community_id}"
        self.members: Set[str] = set()
        self.centroid_interests: Optional[np.ndarray] = None
        self.shared_memes: List[str] = []
        self.internal_language: Dict[str, int] = {}  # word -> frequency
        self.dominant_personality: str = ""
        self.interaction_density = 0.0
        self.polarization_score = 0.0
        self.formation_tick = 0
        self.size_history: List[int] = []
    
    def add_member(self, agent_id: str):
        self.members.add(agent_id)
    
    def remove_member(self, agent_id: str):
        self.members.discard(agent_id)
    
    def update_centroid(self, agent_map: Dict[str, UserAgent]):
        """Update community centroid based on member interests."""
        if not self.members:
            return
        
        vectors = []
        for aid in self.members:
            if aid in agent_map:
                vectors.append(agent_map[aid].interests)
        
        if vectors:
            self.centroid_interests = np.mean(vectors, axis=0)
            norm = np.linalg.norm(self.centroid_interests)
            if norm > 0:
                self.centroid_interests = self.centroid_interests / norm
    
    def compute_polarization(self, agent_map: Dict[str, UserAgent]) -> float:
        """
        Compute community polarization score.
        High polarization = members have very similar interests (echo chamber).
        """
        if len(self.members) < 2:
            return 0.0
        
        vectors = []
        for aid in self.members:
            if aid in agent_map:
                vectors.append(agent_map[aid].interests)
        
        if len(vectors) < 2:
            return 0.0
        
        vectors = np.array(vectors)
        centroid = np.mean(vectors, axis=0)
        
        # Average cosine similarity to centroid
        norms = np.linalg.norm(vectors, axis=1)
        centroid_norm = np.linalg.norm(centroid)
        
        if centroid_norm < 1e-8:
            return 0.0
        
        similarities = vectors @ centroid / (norms * centroid_norm + 1e-8)
        avg_similarity = np.mean(similarities)
        
        # High similarity = high polarization (echo chamber)
        self.polarization_score = float(np.clip(avg_similarity, 0, 1))
        return self.polarization_score


class CommunityDetector:
    """
    Detects and manages communities in the ecosystem.
    
    Uses interest-based clustering and interaction patterns.
    """
    
    def __init__(self, n_topics: int = 20, 
                 merge_threshold: float = 0.7,
                 split_threshold: float = 0.3,
                 min_community_size: int = 3):
        self.n_topics = n_topics
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.min_community_size = min_community_size
        
        self.communities: Dict[int, Community] = {}
        self.next_community_id = 0
        self.community_history: List[Dict] = []
    
    def detect_communities(self, agents: List[UserAgent], 
                           interaction_graph: Dict[str, Set[str]] = None) -> Dict[int, Community]:
        """
        Detect communities using interest vectors and optionally interaction patterns.
        Uses a simple agglomerative clustering approach.
        """
        if not agents:
            return self.communities
        
        # Build agent interest matrix
        agent_ids = [a.id for a in agents]
        interest_matrix = np.array([a.interests for a in agents])
        
        # Compute similarity matrix
        norms = np.linalg.norm(interest_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = interest_matrix / norms
        similarity_matrix = normalized @ normalized.T
        
        # Boost similarity with interaction graph
        if interaction_graph:
            for i, aid_i in enumerate(agent_ids):
                for j, aid_j in enumerate(agent_ids):
                    if aid_j in interaction_graph.get(aid_i, set()):
                        similarity_matrix[i, j] = min(similarity_matrix[i, j] + 0.2, 1.0)
        
        # Simple clustering: assign agents to communities
        n = len(agents)
        assignments = [-1] * n
        
        community_id = 0
        
        for i in range(n):
            if assignments[i] >= 0:
                continue
            
            # Find similar agents
            similar = []
            for j in range(n):
                if i != j and similarity_matrix[i, j] > self.merge_threshold:
                    similar.append(j)
            
            if len(similar) >= self.min_community_size - 1:
                # Create or assign to community
                assignments[i] = community_id
                for j in similar:
                    if assignments[j] < 0:
                        assignments[j] = community_id
                
                community_id += 1
            else:
                # Lone agent - try to find nearest existing community
                if self.communities:
                    best_community = self._find_nearest_community(agents[i], interest_matrix[i])
                    if best_community is not None:
                        assignments[i] = best_community
        
        # Build communities
        new_communities: Dict[int, Community] = {}
        agent_map = {a.id: a for a in agents}
        
        for i, cid in enumerate(assignments):
            if cid < 0:
                continue
            
            if cid not in new_communities:
                new_communities[cid] = Community(cid)
            
            new_communities[cid].add_member(agents[i].id)
            agents[i].community_id = cid
        
        # Merge small communities
        merged = self._merge_small_communities(new_communities, agent_map)
        
        # Update centroids and polarization
        for community in merged.values():
            community.update_centroid(agent_map)
            community.compute_polarization(agent_map)
            community.size_history.append(len(community.members))
        
        # Preserve community names from previous detection
        for cid, community in merged.items():
            if cid in self.communities:
                community.name = self.communities[cid].name
                community.formation_tick = self.communities[cid].formation_tick
                community.shared_memes = self.communities[cid].shared_memes
                community.internal_language = self.communities[cid].internal_language
        
        self.communities = merged
        self.next_community_id = max(self.communities.keys(), default=-1) + 1
        
        return self.communities
    
    def _find_nearest_community(self, agent: UserAgent, agent_vector: np.ndarray) -> Optional[int]:
        """Find the nearest existing community for an agent."""
        best_cid = None
        best_sim = 0.0
        
        for cid, community in self.communities.items():
            if community.centroid_interests is None:
                continue
            
            sim = np.dot(agent_vector, community.centroid_interests)
            if sim > best_sim and sim > self.split_threshold:
                best_sim = sim
                best_cid = cid
        
        return best_cid
    
    def _merge_small_communities(self, communities: Dict[int, Community],
                                  agent_map: Dict[str, UserAgent]) -> Dict[int, Community]:
        """Merge communities that are too small or too similar."""
        if len(communities) <= 1:
            return communities
        
        # Find small communities
        small_cids = [cid for cid, c in communities.items() 
                      if len(c.members) < self.min_community_size]
        
        # Merge small into nearest large
        for cid in small_cids:
            small = communities[cid]
            if not small.members:
                continue
            
            # Find nearest large community
            best_target = None
            best_sim = 0.0
            
            for target_cid, target in communities.items():
                if target_cid == cid or len(target.members) < self.min_community_size:
                    continue
                if target.centroid_interests is not None and small.centroid_interests is not None:
                    sim = np.dot(target.centroid_interests, small.centroid_interests)
                    if sim > best_sim:
                        best_sim = sim
                        best_target = target_cid
            
            if best_target is not None:
                for member_id in small.members:
                    communities[best_target].add_member(member_id)
                    if member_id in agent_map:
                        agent_map[member_id].community_id = best_target
                del communities[cid]
        
        return communities
    
    def compute_ecosystem_polarization(self) -> float:
        """
        Compute overall ecosystem polarization index.
        Higher = more polarized (communities are far apart).
        """
        if len(self.communities) < 2:
            return 0.0
        
        centroids = []
        for community in self.communities.values():
            if community.centroid_interests is not None:
                centroids.append(community.centroid_interests)
        
        if len(centroids) < 2:
            return 0.0
        
        centroids = np.array(centroids)
        sim_matrix = centroids @ centroids.T
        
        # Average inter-community distance (1 - similarity)
        n = len(centroids)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += 1 - sim_matrix[i, j]
                count += 1
        
        return total_dist / count if count > 0 else 0.0
    
    def get_community_stats(self) -> List[Dict]:
        """Get statistics for all communities."""
        stats = []
        for cid, community in self.communities.items():
            stats.append({
                "id": cid,
                "name": community.name,
                "size": len(community.members),
                "polarization": community.polarization_score,
                "shared_memes_count": len(community.shared_memes),
                "vocabulary_size": len(community.internal_language),
            })
        return stats
    
    def record_history(self, tick: int):
        """Record community state for this tick."""
        self.community_history.append({
            "tick": tick,
            "num_communities": len(self.communities),
            "sizes": {cid: len(c.members) for cid, c in self.communities.items()},
            "polarization": {cid: c.polarization_score for cid, c in self.communities.items()},
        })
    
    def to_dict(self) -> dict:
        return {
            "num_communities": len(self.communities),
            "communities": self.get_community_stats(),
            "ecosystem_polarization": self.compute_ecosystem_polarization(),
        }
