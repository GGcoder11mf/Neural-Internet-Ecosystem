"""Core modules for the Fake Internet Ecosystem simulation."""
from .agent import UserAgent, BotAgent, PlatformAgent, PersonalityType
from .content import Post, Meme, Idea, ContentType, generate_post_text, distort_repost_text, TOPIC_LABELS
from .attention import AttentionEngine
from .meme_evolution import MemeEvolver
from .community import CommunityDetector
from .language_drift import LanguageDriftEngine
from .metrics import MetricsTracker
from .simulation import Simulation
from .meme_generator import generate_meme_image, generate_post_image
