"""
Neural network module for the Fake Internet Ecosystem.

A simple CUDA-accelerated feedforward neural network that learns to predict
engagement scores from post features. It trains online as the simulation runs
and augments feed ranking with learned predictions.

Architecture:
    Input:  [topic_vector(n_topics) + author_features(4) + content_features(6)] = n_topics + 10
    Hidden: 128 -> 64 -> 32  (non-narrow, decent width)
    Output: 1 (predicted engagement score)
"""

import numpy as np
import os

# Try importing torch, fall back gracefully
TORCH_AVAILABLE = False
torch = None
nn = None
optim = None
CUDA_AVAILABLE = False
DEVICE = None

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.optim as _optim
    torch = _torch
    nn = _nn
    optim = _optim
    TORCH_AVAILABLE = True
    # Check for CUDA
    CUDA_AVAILABLE = _torch.cuda.is_available()
    DEVICE = _torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    pass


if TORCH_AVAILABLE and nn is not None:
    class FeedRankingNet(nn.Module):
        """
        A feedforward neural network for predicting post engagement.
        
        Non-narrow architecture: 128-64-32 hidden layers with ReLU activations.
        Takes post features (topic vector + author stats + content metrics)
        and predicts an engagement score that augments feed ranking.
        """
        
        def __init__(self, input_dim: int = 30):
            super().__init__()
            self.input_dim = input_dim
            
            # Non-narrow architecture with decent width
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output 0-1 engagement prediction
            )
            
            self.to(DEVICE)
        
        def forward(self, x):
            return self.net(x)
else:
    class FeedRankingNet:
        """Stub class when PyTorch is not available."""
        def __init__(self, input_dim: int = 30):
            self.input_dim = input_dim


class NeuralRankingEngine:
    """
    Manages the neural network for feed ranking.
    
    Trains online as the simulation runs — each tick provides new training data.
    The network learns patterns like:
    - What topics get more engagement
    - How author features affect virality
    - Controversy and novelty interactions
    
    Used to augment the traditional feed ranking with learned predictions.
    """
    
    def __init__(self, n_topics: int = 20, learning_rate: float = 0.001,
                 batch_size: int = 32, train_interval: int = 5):
        self.n_topics = n_topics
        self.input_dim = n_topics + 10  # topics + author + content features
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.tick_count = 0
        
        if TORCH_AVAILABLE:
            self.model = FeedRankingNet(self.input_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
            self.model.train()
        else:
            self.model = None
            self.optimizer = None
            self.criterion = None
        
        # Training data buffer
        self.max_buffer = 500
        self.features_buffer = []
        self.targets_buffer = []
        
        # Stats
        self.train_loss_history = []
        self.prediction_count = 0
    
    def extract_features(self, post, author=None) -> np.ndarray:
        """
        Extract feature vector from a post for neural network input.
        
        Features:
        - topic_vector: n_topics dimensions
        - author_features: [follower_count_norm, is_bot, personality_encoding, posts_created_norm]
        - content_features: [novelty, generation, is_meme, has_hashtags, text_length_norm, tick_age_norm]
        """
        features = np.zeros(self.input_dim, dtype=np.float32)
        
        # Topic vector
        topic = post.topic_vector
        features[:self.n_topics] = topic[:self.n_topics]
        
        # Author features
        offset = self.n_topics
        if author:
            features[offset] = min(len(author.followers) / 50.0, 1.0)  # normalized followers
            features[offset + 1] = 1.0 if getattr(author, 'is_bot', False) else 0.0
            # Personality encoding (simple hash-based)
            personality_val = hash(getattr(author, 'personality', '')) % 10 / 10.0
            features[offset + 2] = personality_val
            features[offset + 3] = min(getattr(author, 'posts_created', 0) / 20.0, 1.0)
        
        # Content features
        features[offset + 4] = getattr(post, 'novelty_score', 1.0)
        features[offset + 5] = min(getattr(post, 'generation', 0) / 5.0, 1.0)
        features[offset + 6] = 1.0 if isinstance(post, type(post)) and post.content_type.value == 'meme' else 0.0
        features[offset + 7] = 1.0 if getattr(post, 'hashtags', []) else 0.0
        features[offset + 8] = min(len(getattr(post, 'text', '')) / 200.0, 1.0)
        features[offset + 9] = min(getattr(post, 'attention_score', 0) / 10.0, 1.0)
        
        return features
    
    def add_training_sample(self, post, author, engagement_score: float):
        """Add a training sample to the buffer."""
        features = self.extract_features(post, author)
        target = np.clip(engagement_score, 0, 1)
        
        self.features_buffer.append(features)
        self.targets_buffer.append(target)
        
        # Trim buffer
        if len(self.features_buffer) > self.max_buffer:
            self.features_buffer = self.features_buffer[-self.max_buffer:]
            self.targets_buffer = self.targets_buffer[-self.max_buffer:]
    
    def train_step(self) -> float:
        """
        Run one training step on buffered data.
        Returns the loss value.
        """
        if not TORCH_AVAILABLE or not self.model:
            return 0.0
        
        if len(self.features_buffer) < self.batch_size:
            return 0.0
        
        self.model.train()
        
        # Sample a batch
        indices = np.random.choice(len(self.features_buffer), self.batch_size, replace=False)
        batch_features = np.array([self.features_buffer[i] for i in indices])
        batch_targets = np.array([self.targets_buffer[i] for i in indices])
        
        # Convert to tensors
        x = torch.FloatTensor(batch_features).to(DEVICE)
        y = torch.FloatTensor(batch_targets).unsqueeze(1).to(DEVICE)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        loss_val = loss.item()
        self.train_loss_history.append(loss_val)
        
        return loss_val
    
    def predict(self, post, author=None) -> float:
        """
        Predict engagement score for a post.
        Returns a value between 0 and 1.
        """
        if not TORCH_AVAILABLE or not self.model:
            return 0.5  # neutral prediction without model
        
        self.model.eval()
        
        features = self.extract_features(post, author)
        x = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = self.model(x)
        
        self.prediction_count += 1
        return float(pred.item())
    
    def predict_batch(self, posts, author_map=None) -> np.ndarray:
        """
        Predict engagement scores for a batch of posts.
        Returns array of predictions between 0 and 1.
        """
        if not TORCH_AVAILABLE or not self.model:
            return np.full(len(posts), 0.5)
        
        if not posts:
            return np.array([])
        
        self.model.eval()
        
        features = []
        for post in posts:
            author = author_map.get(post.author_id) if author_map else None
            features.append(self.extract_features(post, author))
        
        x = torch.FloatTensor(np.array(features)).to(DEVICE)
        
        with torch.no_grad():
            pred = self.model(x)
        
        return pred.cpu().numpy().flatten()
    
    def should_train(self) -> bool:
        """Check if it's time to train."""
        self.tick_count += 1
        return self.tick_count % self.train_interval == 0
    
    def get_status(self) -> dict:
        """Get neural network status for display."""
        status = {
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": CUDA_AVAILABLE,
            "device": str(DEVICE) if DEVICE else "N/A",
            "model_loaded": self.model is not None,
            "buffer_size": len(self.features_buffer),
            "prediction_count": self.prediction_count,
            "train_steps": len(self.train_loss_history),
        }
        if self.train_loss_history:
            status["latest_loss"] = round(self.train_loss_history[-1], 4)
            status["avg_loss"] = round(np.mean(self.train_loss_history[-20:]), 4)
        return status
