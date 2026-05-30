# 🌐 FakeNet Ecosystem — AI-Powered Internet Simulation

**A rich, agent-based simulation of a fake social media ecosystem with emergent communities, meme evolution, language drift, neural network feed ranking, and real-time visualization.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Core Simulation Engine](#core-simulation-engine)
  - [Agent System](#agent-system)
  - [Content System](#content-system)
  - [Attention Engine](#attention-engine)
  - [Meme Evolution](#meme-evolution)
  - [Community Detection](#community-detection)
  - [Language Drift](#language-drift)
  - [Platform Algorithm](#platform-algorithm)
  - [Neural Network Feed Ranking](#neural-network-feed-ranking)
- [Meme Image Generator](#meme-image-generator)
- [Metrics & Analytics](#metrics--analytics)
- [Web Interface](#web-interface)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration Parameters](#configuration-parameters)
- [How It Works — Simulation Tick Lifecycle](#how-it-works--simulation-tick-lifecycle)
- [Emergent Phenomena](#emergent-phenomena)
- [Technology Stack](#technology-stack)
- [License](#license)

---

## Overview

FakeNet Ecosystem is a sophisticated agent-based simulation that models the dynamics of a social media platform. It spawns AI-driven agents — each with unique personalities, interests, and behavioral patterns — who create, consume, like, repost, remix, and argue over content in a simulated feed. Over time, communities form, memes evolve through mutation and merging, language drifts as communities develop their own dialects, information gets distorted through reposts (like a digital telephone game), and a neural network learns to rank content based on observed engagement patterns.

The simulation runs in discrete **ticks**, each representing a "moment" in internet time. The entire system is served through a Twitter-inspired dark-themed web UI with real-time Server-Sent Events (SSE) updates, interactive controls, and Chart.js-powered analytics dashboards.

---

## Key Features

| Feature | Description |
|---|---|
| **10 Personality Types** | Curious, Toxic, Viral Chaser, Niche Thinker, Lurker, Influencer, Meme Lord, Echo Seeker, Doom Scroller, Hot Taker — each with deeply tuned behavioral parameters |
| **Bot Agents** | Amplifier, Spammer, and Coordinator bots that simulate inauthentic behavior, coordinated amplification, and engagement bait |
| **Meme Evolution** | Memes mutate (shorten, shift, slang, distort), merge, and undergo natural selection under attention pressure — tracking full lineage/family trees |
| **Information Distortion** | The telephone game: reposted content gets exaggerated, simplified, twisted, misquoted, or condensed over relay chains |
| **Community Detection** | Interest-based agglomerative clustering with merge/split dynamics, polarization scoring, and echo chamber detection |
| **Language Drift** | Communities develop their own dialects through word compression, emoji replacement, grammar simplification, and emergent slang generation |
| **Neural Network Ranking** | PyTorch feedforward network (128→64→32) that trains online during simulation and augments feed ranking with learned engagement predictions |
| **CUDA Acceleration** | GPU-accelerated attention matrix, similarity, and feed ranking computations via PyTorch with automatic NumPy fallback |
| **Meme Image Generation** | PIL/Pillow-powered procedural image generation with personality-specific color schemes, background patterns, and visual evolution indicators |
| **Real-Time UI** | Server-Sent Events (SSE) for live tick updates, auto-tick mode with configurable speed, Twitter-inspired dark theme |
| **Rich Analytics** | Gini coefficient (attention inequality), ecosystem polarization, language divergence, controversy/distortion tracking, influencer dominance, personality distributions — all plotted over time with Chart.js |
| **Sentiment Waves** | Random collective sentiment events (positive, negative, controversial) that sweep through the ecosystem and affect agent emotional states |
| **Follow Cascades** | When influencers follow someone, their followers may cascade-follow, modeling real influencer-driven network growth |
| **Circadian Rhythms** | Each agent has a unique activity phase and amplitude, simulating timezone-like sleep/wake cycles |
| **Doom Scrolling** | Agents with high doom-scroll susceptibility get trapped reading negative content, with the algorithm feeding the spiral |
| **Controversy & Ratio Detection** | Posts with replies far exceeding likes get flagged as controversial or "ratio'd," modeling real social media dynamics |

---

## Architecture

The project follows a modular architecture with a clear separation between the simulation engine, GPU computation layer, web server, and frontend:

```
┌──────────────────────────────────────────────────────────┐
│                    Flask Web Server                       │
│  (server.py — routes, SSE, API endpoints)                │
├──────────────┬───────────────────────┬───────────────────┤
│              │                       │                   │
│  ┌───────────▼──────────┐  ┌────────▼────────┐  ┌──────▼──────┐
│  │   Core Simulation    │  │  GPU Backend    │  │  Frontend   │
│  │                      │  │                 │  │             │
│  │  • Agent System      │  │  • PyTorch NN   │  │  • HTML/JS  │
│  │  • Content System    │  │  • CUDA Tensors │  │  • CSS      │
│  │  • Attention Engine  │  │  • NumPy Fallback│  │  • Chart.js │
│  │  • Meme Evolution    │  │                 │  │  • SSE      │
│  │  • Community Detect  │  └─────────────────┘  └─────────────┘
│  │  • Language Drift    │
│  │  • Metrics Tracker   │
│  │  • Meme Generator    │
│  └──────────────────────┘
└──────────────────────────────────────────────────────────┘
```

---

## Core Simulation Engine

### Agent System

**Files:** `core/agent.py`

The agent system is the heart of the simulation. Three types of agents inhabit the ecosystem:

#### UserAgent — Human-like AI Individuals

Each `UserAgent` has:

- **Interest Vector**: A Dirichlet-distributed vector over N topics that determines what content the agent cares about. Interests slowly drift as agents consume content (reinforcement learning).
- **Personality Type**: One of 10 archetypes that govern all behavioral parameters.
- **Emotional State**: A floating-point value in [-1, 1] that persists between ticks and influences decision-making. Emotional states decay toward neutral with a momentum factor of 0.92.
- **Memory**: A rolling window of up to 200 (post_id, emotional_score) pairs that prevents agents from repeatedly engaging with the same content.
- **Social Graph**: Following/follower sets that determine feed ranking proximity.
- **Circadian Rhythm**: Each agent has an `activity_phase` (0–1, like a timezone offset) and `activity_amplitude` that modulates their activity level as a cosine wave over a 24-tick "day."
- **Doom Scroll State**: Tracks how deep an agent is in a negative content spiral, with a cooldown mechanism for recovery.
- **Content Exhaustion**: Tracks topic exposure per topic index, decaying at 0.9 per tick — agents get fatigued from seeing the same topics repeatedly.
- **Curiosity Cycles**: A sinusoidal curiosity phase that periodically drives agents to explore new topics.
- **Pile-On Tracking**: Toxic/Hot Taker agents remember which controversial posts they've already piled onto.

#### Personality Types & Parameters

| Personality | Novelty Weight | Engagement Weight | Echo Weight | Post Prob | Like Prob | Controversy Attraction | Doom Scroll Susceptibility |
|---|---|---|---|---|---|---|---|
| **Curious** | 0.8 | 0.3 | 0.1 | 0.25 | 0.35 | 0.2 | 0.15 |
| **Toxic** | 0.3 | 0.8 | 0.2 | 0.40 | 0.15 | 0.9 | 0.6 |
| **Viral Chaser** | 0.2 | 0.9 | 0.15 | 0.35 | 0.25 | 0.6 | 0.3 |
| **Niche Thinker** | 0.5 | 0.3 | 0.6 | 0.15 | 0.4 | 0.15 | 0.1 |
| **Lurker** | 0.5 | 0.2 | 0.2 | 0.03 | 0.5 | 0.3 | 0.45 |
| **Influencer** | 0.4 | 0.7 | 0.2 | 0.50 | 0.1 | 0.5 | 0.2 |
| **Meme Lord** | 0.7 | 0.5 | 0.15 | 0.40 | 0.3 | 0.35 | 0.2 |
| **Echo Seeker** | 0.1 | 0.4 | 0.9 | 0.20 | 0.5 | 0.4 | 0.35 |
| **Doom Scroller** | 0.3 | 0.6 | 0.3 | 0.08 | 0.2 | 0.7 | 0.9 |
| **Hot Taker** | 0.4 | 0.8 | 0.15 | 0.45 | 0.08 | 0.95 | 0.3 |

#### Decision Logic — `decide_action()`

When an agent encounters a post, they run through a priority-based decision tree:

1. **Meme Lords** check for remix opportunity first (high mutation probability)
2. **Toxic/Hot Taker** agents check for pile-on opportunity on controversial posts
3. **Like** — the most common action, probability scaled by emotional score and activity level
4. **Reply** — driven by emotional intensity, boosted by controversy
5. **Repost** — driven by sharing desire, Viral Chasers repost trending content 2x more
6. **Read** — default passive consumption

Emotional scores determine whether an agent even considers engaging (scores below -0.3 usually lead to ignoring, unless the agent is a doom scroller). Each engagement action slightly shifts the agent's interests toward the post's topic vector.

#### BotAgent — Automated Accounts

Three bot types with distinct behaviors:

- **Amplifier**: Retweets content matching specific target topics with 60% probability. Posts occasionally to appear legitimate.
- **Spammer**: Posts repetitive content on a fixed interval (3–8 ticks). Rarely interacts with others' content (5% like rate). Uses `bot_spam` templates with engagement bait phrasing.
- **Coordinator**: Amplifies in coordinated waves, synchronized by `coordination_delay` and group ID. Posts on a fixed schedule with 30% probability.

Bot names are procedurally generated (e.g., `crypto_bot`, `TrendingHq`, `news3847`) to mimic real bot naming patterns.

---

### Content System

**Files:** `core/content.py`

#### Content Types

| Type | Description |
|---|---|
| `TEXT` | Standard text post |
| `IMAGE` | Post with an attached image |
| `IDEA` | Compressed meaning vector content |
| `MEME` | Special post type that can mutate and evolve |
| `QUOTE` | Quote tweet / repost with commentary |
| `THREAD` | Part of a conversation thread |

#### Post Object

Every post tracks:

- **Topic Vector**: A normalized vector over N topics representing the post's semantic content
- **Engagement Metrics**: Views, likes, reposts, replies, ignores, quotes
- **Controversy Score**: Computed from the reply-to-like ratio (high replies + quotes relative to likes = controversial)
- **Ratio Score**: `(replies + quotes) / likes` — when this exceeds 2.0, the post is considered "ratio'd"
- **Distortion Level**: Cumulative measure of how much the content has been distorted from its original form through reposts
- **Novelty Score**: Decays exponentially over time (`novelty_decay ^ age`)
- **Attention Score**: Computed by the Attention Engine each tick
- **Mutation History**: List of all mutations this content has undergone (for meme lineage tracking)
- **Generation**: 0 = original, 1+ = remix/mutation generation
- **Thread Tracking**: `thread_id` and `thread_position` for conversation threading

#### Meme Mutation Types

| Mutation | Effect |
|---|---|
| `shorten` | Removes 1–3 random words from the text |
| `shift` | Adds Gaussian noise to the topic vector |
| `slang` | Prepends or appends internet slang (e.g., "lol", "ngl", "based", "no cap", "slay", "mid") |
| `distort` | Telephone game — exaggerates, simplifies, twists, misquotes, or condenses the text |
| `merge` | Combines text and topic vectors from two parent memes |

The **distort** mutation is particularly rich, with five sub-types:

- **Exaggerate**: Inserts intensifiers ("literally", "insanely", "unbelievably")
- **Simplify**: Removes nuance words ("maybe", "perhaps", "possibly")
- **Twist**: Flips positive/negative words ("good" → "terrible", "love" → "hate")
- **Misquote**: Replaces a key word with "apparently", "supposedly", "sources say"
- **Condense**: Truncates to the first half, losing context

Memes can also **merge** — combining text and topic vectors from two parent memes into a new child, creating recombinant content.

#### Post Generation — 100+ Templates

Posts are generated from personality-driven template pools with dynamic placeholders:

- `{topic}`: Dominant topic label (e.g., "tech", "politics", "gaming")
- `{topic2}`: Secondary topic for cross-topic posts
- `{adj}`: Random adjective (positive or neutral)
- `{contrary_adj}`: Contrarian adjective ("overrated", "underrated", "mid", "a scam")
- `{year}`: Random recent year

Emoji usage varies by personality (Meme Lords at 60%, Niche Thinkers at 5%). Hashtags are generated from topic vectors with personality-weighted usage rates (Influencers at 65%, Lurkers at 10%).

#### Information Distortion on Repost

When a user reposts content, there's a 15% chance the text gets distorted through five mechanisms:

1. **Drop word**: Information loss — a random word is removed
2. **Add filler**: Dilution — "like", "basically", "literally" is inserted
3. **Swap similar**: Semantic shift — "some" → "most", "could" → "will", "suggests" → "proves"
4. **Truncate**: Context loss — text is cut short
5. **Add hedge**: Softening — "apparently", "supposedly", "some say" is prepended

---

### Attention Engine

**Files:** `core/attention.py`

Attention is the core resource that flows through the ecosystem — like electricity through a circuit. The Attention Engine computes how attention is allocated, distributed, and decayed.

#### Attention Formula

```
attention(post) = log1p(views) × (1 + engagement_rate) × novelty_decay^age × controversy_boost × engagement_bait_boost
```

Where:

- **Controversy Boost**: When controversy_score > 0.3, attention is multiplied by `1 + (controversy - 0.3) × controversy_amplification` — modeling how platforms amplify controversial content because engagement = revenue.
- **Engagement Bait Boost**: When ratio_score > 2.0 (replies >> likes), attention is multiplied by `1 + engagement_bait_weight × min(ratio, 10)` — provocative content that drives argument gets boosted.

#### Feed Visibility

For each agent-post pair, visibility is computed as:

```
visibility = attention × (1 + platform_bias × distance_factor × echo_chamber) × (1 + doom_factor × 0.3)
```

Where `distance_factor = 1 / (1 + network_distance)`, and `doom_factor` amplifies visibility of negative content for agents stuck in doom scrolls.

#### Network Distance

Distance between an agent and a post's author is computed as:

| Relationship | Distance |
|---|---|
| Same agent | 0.0 |
| Direct follow | 0.2 |
| 2nd degree (follow of follow) | 0.5 |
| Similar interests | `1 - cosine_similarity` |
| Unknown | 1.0 |

#### Attention Inequality — Gini Coefficient

The engine tracks the Gini coefficient of attention distribution, analogous to wealth inequality but for views. A Gini near 1 means a few posts dominate all attention; near 0 means attention is evenly distributed.

#### Virality Cascade Model

Virality is modeled using an SIR-like (Susceptible-Infected-Recovered) cascade:

```
cascade_prob = (base_engagement_rate × 0.1 + novelty × 0.05) × saturation_factor × controversy_multiplier
```

The saturation factor (`1 / (1 + log1p(n_exposed))`) models how each additional exposure is less impactful. Cascade probability is capped at 30% per exposure.

---

### Meme Evolution

**Files:** `core/meme_evolution.py`

The `MemeEvolver` manages the evolutionary lifecycle of memes across the ecosystem, treating meme spread as a form of genetic drift under attention pressure.

#### Key Concepts

- **Meme Families**: Lineage trees tracked from root meme to all descendants. Each family has a root ID and a list of descendant IDs.
- **Mutation Rate**: Base rate of 0.3, boosted by personality (Meme Lords get +0.5, Toxic +0.15).
- **Merge Probability**: 10% chance that two memes combine when a Meme Lord remixes.
- **Selection Pressure**: Memes below `median_attention × (1 - selection_pressure)` may go extinct.

#### Evolution Process

1. **Should Mutate?** — Based on mutation rate + personality modifier
2. **Should Merge?** — If two memes are available and merge probability triggers
3. **Pick Mutation Type** — Weighted: shift (35%), slang (35%), shorten (30%)
4. **Apply Mutation** — Creates a child meme with incremented generation and updated distortion level
5. **Register in Lineage** — Walk up parent chain to find root, add to family tree
6. **Apply Selection Pressure** — Low-attention memes are marked as potentially extinct

#### Meme Diversity

Diversity is measured as `1 - mean(pairwise_cosine_similarity)` across all meme topic vectors, sampled to 100 memes if there are too many.

---

### Community Detection

**Files:** `core/community.py`

Communities form naturally as agents with similar interests cluster together, reinforced by the platform's echo chamber algorithm.

#### Detection Algorithm

1. Build a cosine similarity matrix over all agent interest vectors
2. Boost similarity by +0.2 for agents who interact with each other
3. Group agents with similarity > `merge_threshold` (0.7) into communities
4. Lone agents are assigned to the nearest existing community
5. Small communities (< `min_community_size` = 3) are merged into the nearest large community
6. Centroids and polarization scores are updated

#### Community Object

Each community tracks:

- **Members**: Set of agent IDs
- **Centroid Interests**: Mean interest vector of all members (normalized)
- **Shared Memes**: Memes circulating within the community
- **Internal Language**: Community-specific word frequency map
- **Polarization Score**: Average cosine similarity of members to the centroid. High similarity = high polarization (echo chamber).
- **Interaction Density**: How much members interact with each other vs. outsiders

#### Ecosystem Polarization

The overall ecosystem polarization is computed as the average inter-community distance (1 - cosine similarity between centroids). Higher values mean communities are far apart in interest space.

---

### Language Drift

**Files:** `core/language_drift.py`

Over time, each community develops its own linguistic identity through four mechanisms:

1. **Word Compression**: "because" → "cuz", "probably" → "prolly", "going to" → "gonna", "for real" → "fr", "not gonna lie" → "ngl" (48 compression rules)
2. **Emoji Replacement**: "laughing" → 😂, "fire" → 🔥, "love" → ❤️, "skull" → 💀 (20 emoji rules)
3. **Grammar Simplification**: Randomly removes articles and copula verbs ("a", "an", "the", "is", "are", "was", "were")
4. **Emergent Slang Generation**: Communities corrupt frequent words into new slang through:
   - **Shorten**: Take first half of the word
   - **Repeat**: Double a random letter
   - **Replace**: Swap vowels randomly
   - **Combine**: Blend word with a slang pool term

The engine tracks a global vocabulary (Counter), per-community vocabularies, community-specific dialect mappings (standard → dialect), and adopted slang sets.

#### Language Divergence

Divergence between communities is measured using Jaccard distance of their top 30 words:

```
divergence = 1 - (|intersection(top_words_A, top_words_B)| / |union(top_words_A, top_words_B)|)
```

This is computed for every pair of communities and averaged. Over time, this metric tracks how linguistically isolated communities become.

---

### Platform Algorithm

**Files:** `core/agent.py` (`PlatformAgent` class)

The `PlatformAgent` models the social media platform itself — the algorithm that ranks feeds, boosts content, and evolves over time.

#### Platform Behaviors

- **Trending Detection**: Posts exceeding the attention threshold (5.0) are marked as trending (max 20)
- **Topic Boosting/Suppression**: Platform-level boost and suppress vectors that affect all posts
- **Recommendation Bias**: Content with existing engagement gets a logarithmic boost (`log1p(engagement) × recommendation_bias`)
- **Controversy Boost**: Controversial content gets amplified — and this parameter **slowly increases over time** (0.02 increments, modeling how platforms optimize for engagement)
- **Algorithm Evolution**: Every tick, there's a 0.1% chance of mutating `echo_chamber_strength` or `virality_sensitivity` by ±0.05. Every 50 ticks, boosted topics are reshuffled.

This models the real-world dynamic where platform algorithms gradually become more engagement-optimized, creating stronger echo chambers and amplifying controversy.

---

### Neural Network Feed Ranking

**Files:** `gpu/neural_net.py`

A PyTorch feedforward neural network trains online during the simulation and augments the traditional feed ranking with learned engagement predictions.

#### Architecture

```
Input (n_topics + 10) → Linear(128) → ReLU → Dropout(0.1) → Linear(64) → ReLU → Dropout(0.1) → Linear(32) → ReLU → Linear(1) → Sigmoid
```

**Input Features** (n_topics + 10 dimensions):

| Feature | Dimensions | Description |
|---|---|---|
| Topic Vector | n_topics | Post's topic distribution |
| Author Followers | 1 | Normalized follower count (0–1) |
| Is Bot | 1 | Binary bot indicator |
| Personality Encoding | 1 | Hash-based personality value (0–1) |
| Posts Created | 1 | Normalized post count (0–1) |
| Novelty Score | 1 | Current novelty of the post |
| Generation | 1 | Mutation generation (0–1) |
| Is Meme | 1 | Binary meme indicator |
| Has Hashtags | 1 | Binary hashtag presence |
| Text Length | 1 | Normalized text length (0–1) |
| Attention Score | 1 | Current attention score (0–1) |

#### Training

- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE (predicting engagement score)
- **Batch Size**: 32
- **Buffer**: Rolling window of 500 training samples (features + targets)
- **Train Interval**: Every 5 ticks
- **Online Learning**: Each like, repost, or remix action adds a training sample

#### Feed Ranking Integration

Neural network predictions are blended with traditional feed ranking:

```
feed_score = attention × virality + interest_similarity × echo_chamber + platform_bias × proximity + nn_prediction × 0.3
```

The neural network weight of 0.3 provides learned signal without overpowering the handcrafted ranking features. The system gracefully degrades when PyTorch is unavailable (returns neutral 0.5 predictions).

---

## Meme Image Generator

**Files:** `core/meme_generator.py`

Each meme (and some posts) gets a procedurally generated image using PIL/Pillow. Images are personality-themed with distinct visual identities:

#### Color Schemes

| Personality | Background | Accent | Text Color | Vibe |
|---|---|---|---|---|
| Curious | Cool blues | Light blue | Icy white | Inquisitive |
| Toxic | Dark reds | Hot red | Pink-white | Aggressive |
| Viral Chaser | Purple-magenta | Yellow-gold | Warm white | Flashy |
| Niche Thinker | Deep purple | Electric purple | Lavender | Intellectual |
| Lurker | Grey | Muted grey | Silver | Subtle |
| Influencer | Rose-gold | Gold | Warm white | Glamorous |
| Meme Lord | Dark green | Neon green | Lime-white | Chaotic |
| Echo Seeker | Purple-pink | Hot pink | Pink-white | Reflective |
| Doom Scroller | Dark navy | Dark blue | Pale blue | Ominous |
| Hot Taker | Burnt orange | Orange-red | Warm cream | Fiery |

#### Background Patterns

Four procedural background types (selected by hashing the post ID):
- **Gradient**: Smooth vertical color interpolation
- **Noise**: Gradient with random noise overlay for texture
- **Geometric**: Gradient with random rectangles, circles, and lines in accent color
- **Radial**: Concentric circle gradient emanating from center

#### Visual Features

- **Top Label Bar**: Personality-specific label ("THOUGHT", "HOT TAKE", "TRENDING", "MEME", "SPAM", etc.)
- **Generation Indicator**: "GEN N" displayed for mutated memes
- **Text Shadow**: Black shadow behind white text for readability
- **Mutation Type Tags**: Colored badges at the bottom showing mutation types
- **Color Shift**: Each generation shifts the color scheme brighter (up to +60 per channel)
- **Noise Overlay**: Subtle noise for a "realistic" look (5% blend)
- **Smooth Filter**: PIL's SMOOTH filter for antialiasing

Image generation is rate-limited to 5 per tick to manage performance. Old images are cleaned up when exceeding 1000 files.

---

## Metrics & Analytics

**Files:** `core/metrics.py`

The `MetricsTracker` records comprehensive ecosystem metrics every tick:

| Metric | Description |
|---|---|
| `mean_attention` | Average attention score across all posts |
| `max_attention` | Peak attention score |
| `attention_gini` | Gini coefficient of attention distribution (inequality) |
| `attention_entropy` | Shannon entropy of attention distribution |
| `follower_gini` | Gini coefficient of follower distribution |
| `influencer_dominance` | Share of total followers held by top 10% of agents |
| `mean_emotional_state` | Average emotional state across all agents |
| `mean_engagement_rate` | Average post engagement rate |
| `ecosystem_polarization` | Average inter-community distance |
| `language_divergence` | Jaccard distance between community vocabularies |
| `mean_controversy` | Average post controversy score |
| `mean_distortion` | Average information distortion level |
| `bot_post_fraction` | Fraction of posts from bot accounts |
| `ratioed_count` | Number of posts with ratio_score > 2.0 |
| `personality_distribution` | Count of agents per personality type |

---

## Web Interface

**Files:** `server.py`, `templates/`, `static/`

The web UI is a Twitter-inspired dark theme with five main pages:

### Pages

| Page | Route | Description |
|---|---|---|
| **Home** | `/` | Feed view with sorting tabs (Top, Latest, Liked, Spicy). Shows posts with avatars, badges (BOT, MEME, VIRAL, SPICY, RATIO'D, DISTORTED), hashtags, mutation tags, and meme images. |
| **Explore** | `/explore` | Trending topics, top agents by followers, controversial posts, and viral posts. |
| **Communities** | `/communities` | Community cards showing size, polarization bar, member list, dialect mappings, adopted slang, and top words. Ecosystem polarization summary. |
| **Memes** | `/memes` | Meme Lab — top memes by attention with generation badges, mutation type breakdown, family trees, and generated meme images. |
| **Analytics** | `/analytics` | Full dashboard with summary metric cards and Chart.js line/doughnut charts for attention over time, Gini coefficient, community count, language divergence, controversy/distortion, bot content fraction, personality distribution, and influencer dominance. |

### Real-Time Updates

- **Server-Sent Events (SSE)**: The `/api/stream` endpoint pushes tick events to all connected clients, enabling live feed updates when auto-tick is running.
- **Auto-Tick Mode**: Start/stop background tick execution with configurable speed (0.2–10 seconds per tick).
- **Neural Net Status**: Real-time indicator showing CUDA/CPU/Off state with training step count and loss.

### Simulation Controls

The right sidebar provides interactive parameter sliders:

- Number of Agents (10–200)
- Number of Topics (5–50)
- Bot Fraction (0–40%)
- Echo Chamber Strength (0–1)
- Virality Sensitivity (0.1–3.0)
- Novelty Decay (0.80–0.99)
- Mutation Rate (0–1)

---

## API Reference

### Simulation Control

| Endpoint | Method | Description |
|---|---|---|
| `/api/create` | POST | Create a new simulation with parameters |
| `/api/tick` | POST | Run a single tick |
| `/api/ticks` | POST | Run N ticks (`{"n": 10}`) |
| `/api/reset` | POST | Reset and recreate simulation |
| `/api/auto-tick/start` | POST | Start auto-ticking (configurable speed) |
| `/api/auto-tick/stop` | POST | Stop auto-ticking |
| `/api/auto-tick/status` | GET | Check auto-tick status and speed |
| `/api/state` | GET | Full simulation state |
| `/api/stream` | GET | SSE endpoint for real-time updates |

### Data Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/feed` | GET | Post feed (`?sort=attention|recent|engagement|controversial|liked&limit=50`) |
| `/api/trending` | GET | Trending posts and hashtags |
| `/api/agents` | GET | Agent list (`?sort=followers|posts|likes|views&limit=30`) |
| `/api/communities` | GET | Community details with dialects |
| `/api/memes` | GET | Meme list with lineage info |
| `/api/analytics` | GET | Full analytics with time-series history |
| `/api/language` | GET | Language drift data (divergence, vocabularies, dialects) |
| `/api/platform` | GET | Platform algorithm state |
| `/api/neural-net` | GET | Neural network status |

---

## Project Structure

```
fake-internet-ecosystem/
├── server.py                  # Flask web server with API endpoints and SSE
├── requirements.txt           # Python dependencies
├── core/
│   ├── __init__.py           # Module exports
│   ├── agent.py              # UserAgent, BotAgent, PlatformAgent, PersonalityType
│   ├── attention.py          # AttentionEngine — attention scoring and distribution
│   ├── community.py          # Community, CommunityDetector — clustering and polarization
│   ├── content.py            # Post, Meme, Idea, ContentType, templates, distortion
│   ├── language_drift.py     # LanguageDriftEngine — compression, slang, divergence
│   ├── meme_evolution.py     # MemeEvolver — mutation, merge, selection, lineage
│   ├── meme_generator.py     # PIL image generation for memes and posts
│   ├── metrics.py            # MetricsTracker — time-series analytics
│   └── simulation.py         # Simulation — main orchestrator
├── gpu/
│   ├── __init__.py           # Module exports
│   ├── backend.py            # GPU-accelerated computations (PyTorch/NumPy)
│   └── neural_net.py         # FeedRankingNet, NeuralRankingEngine
├── static/
│   ├── css/
│   │   └── style.css         # Twitter-inspired dark theme
│   ├── js/
│   │   └── app.js            # Shared API helpers, controls, SSE, sidebar logic
│   └── memes/                # Generated meme images (created at runtime)
├── templates/
│   ├── base.html             # Base layout with sidebar, controls, trending
│   ├── index.html            # Home feed page
│   ├── explore.html          # Explore trending/agents/viral page
│   ├── communities.html      # Community detection visualization
│   ├── memes.html            # Meme Lab with lineage and images
│   └── analytics.html        # Chart.js analytics dashboard
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements are:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24.0 | Vector math, array operations |
| `flask` | ≥3.0.0 | Web server, API, template rendering |
| `torch` | ≥2.0.0 | Neural network, GPU acceleration (optional) |
| `Pillow` | ≥10.0.0 | Meme image generation |

> **Note**: PyTorch is optional — the simulation runs with NumPy fallbacks if PyTorch is not installed. However, neural network feed ranking will be disabled without it.

### Running the Server

```bash
python server.py
```

The server starts on `http://0.0.0.0:7860` with threading enabled.

### Quick Start

1. Open `http://localhost:7860` in your browser
2. Adjust simulation parameters in the right sidebar
3. Click **🆕 Create** to initialize the simulation
4. Click **▶ Run Tick** to step through manually, or **▶ Live** for real-time auto-ticking
5. Navigate between Home, Explore, Communities, Memes, and Analytics pages

---

## Configuration Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `n_agents` | 50 | 10–200 | Total number of agents (humans + bots) |
| `n_topics` | 20 | 5–50 | Dimensionality of topic space |
| `echo_chamber_strength` | 0.5 | 0–1 | How much similar interests boost feed visibility |
| `virality_sensitivity` | 1.0 | 0.1–3.0 | Multiplier on attention in feed ranking |
| `novelty_decay` | 0.95 | 0.80–0.99 | Per-tick decay rate of content novelty |
| `mutation_rate` | 0.3 | 0–1 | Base probability of meme mutation on repost |
| `max_posts` | 500 | — | Maximum posts retained in memory (pruned by attention) |
| `community_detection_interval` | 5 | — | Ticks between community detection runs |
| `bot_fraction` | 0.1 | 0–0.4 | Fraction of agents that are bots |
| `sentiment_wave_probability` | 0.02 | — | Per-tick probability of a sentiment wave event |
| `generate_images` | True | — | Whether to generate meme/post images |

---

## How It Works — Simulation Tick Lifecycle

Each tick executes the following 14 steps in order:

```
┌─────────────────────────────────────────────────────────────┐
│  TICK N                                                      │
│                                                              │
│  1. Check for sentiment wave trigger                         │
│  2. Apply active sentiment wave effects to agents            │
│  3. Compute attention scores for all posts                   │
│  4. Platform updates trending list & evolves algorithm       │
│  5. Generate personalized feeds & process agent decisions    │
│     └── For each active agent:                               │
│         ├── Rank feed (attention + similarity + NN)          │
│         ├── Decide actions (like/repost/remix/reply/read)    │
│         ├── Apply information distortion on repost           │
│         ├── Evolve memes on remix                            │
│         ├── Process language drift on new text               │
│         ├── Generate meme images                             │
│         └── Maybe create original post                       │
│  6. Add new posts to the ecosystem                           │
│  7. Prune low-attention posts (max_posts limit)              │
│  8. Update social graph (follow/unfollow/cascades)           │
│  9. Decay agent content exhaustion                           │
│  10. Community detection (periodic)                          │
│  11. Language drift — generate community slang               │
│  12. Meme evolution — apply selection pressure               │
│  13. Train neural network (periodic)                         │
│  14. Record all metrics                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Emergent Phenomena

When you run the simulation, you can observe several emergent behaviors that arise from the interaction of simple rules:

1. **Echo Chambers**: Agents with high echo_weight cluster together, reinforce each other's interests, and become increasingly polarized. The platform algorithm's echo_chamber_strength amplifies this effect.

2. **Meme Lineages**: A single meme can spawn dozens of descendants through mutation chains. Some family trees grow large while others go extinct — natural selection in action.

3. **Language Fragmentation**: Communities gradually develop distinct dialects. Words get compressed, slang emerges independently per community, and the language divergence metric increases over time.

4. **Attention Inequality**: The Gini coefficient of attention rises as a few viral posts dominate the ecosystem while the vast majority receive negligible attention — mirroring real social media dynamics.

5. **Controversy Spirals**: Controversial posts get amplified by the platform (engagement = revenue), attracting more replies than likes, which further increases their controversy score — a positive feedback loop.

6. **Information Degradation**: Content that gets relayed through multiple reposts accumulates distortion — nuance is stripped, claims are exaggerated, and meaning shifts. The distortion_level metric tracks this degradation.

7. **Doom Scroll Traps**: Agents with high doom_scroll_susceptibility get caught in negative content spirals. The algorithm feeds them more negative content, deepening their engagement.

8. **Bot Amplification**: Amplifier bots boost specific topics, making them appear more popular than they are. Coordinator bots create waves of synchronized amplification that can push content to trending.

9. **Follow Cascades**: When an influencer follows a new account, a fraction of their followers follow too — modeling real influencer-driven network growth.

10. **Algorithmic Drift**: The platform's controversy_boost parameter slowly increases over time, modeling how platforms gradually optimize for engagement at the cost of content quality.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.8+, Flask 3.0+ |
| **Simulation** | NumPy 1.24+, PyTorch 2.0+ (optional) |
| **Image Generation** | Pillow 10.0+ |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Charts** | Chart.js 4 (CDN) |
| **Real-Time** | Server-Sent Events (SSE) |
| **GPU** | CUDA via PyTorch (with NumPy fallback) |

---

## License

This project is provided as-is for educational and research purposes. The simulation is designed to study emergent phenomena in social media ecosystems, including echo chambers, information distortion, and algorithmic amplification dynamics.
