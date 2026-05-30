"""
Flask server for the Fake Internet Ecosystem.
Serves a Twitter-like UI with real-time simulation updates.

v3: SSE for semi real-time (1 tick/second), meme images, neural net status.
"""

import json
import numpy as np
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.simulation import Simulation
from core.content import Meme, TOPIC_LABELS

app = Flask(__name__, static_folder='static')

# Global simulation state
sim: Simulation = None
sim_lock = threading.Lock()

# Auto-tick state
auto_tick_running = False
auto_tick_thread = None
auto_tick_speed = 1.0  # seconds between ticks

# SSE subscribers
sse_subscribers = []


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def json_response(data, status=200):
    """Helper to return JSON with numpy support."""
    return Response(json.dumps(data, cls=NumpyEncoder), status=status, mimetype='application/json')


def get_sim():
    return sim


def sse_notify(event: str, data: dict):
    """Send an SSE event to all connected clients."""
    msg = f"event: {event}\ndata: {json.dumps(data, cls=NumpyEncoder)}\n\n"
    dead = []
    for i, queue in enumerate(sse_subscribers):
        try:
            queue.put(msg, block=False)
        except Exception:
            dead.append(i)
    for i in reversed(dead):
        sse_subscribers.pop(i)


def auto_tick_loop():
    """Background thread that runs ticks at configured speed."""
    global auto_tick_running
    while auto_tick_running:
        with sim_lock:
            s = get_sim()
            if s is not None:
                try:
                    result = s.run_tick()
                    # Send SSE update
                    sse_notify("tick", {
                        "tick": s.tick,
                        "n_posts": len(s.posts),
                        "n_agents": len(s.agents),
                        "metrics": result,
                    })
                except Exception as e:
                    sse_notify("error", {"message": str(e)})
        time.sleep(auto_tick_speed)


# ===================== PAGES =====================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/explore")
def explore():
    return render_template("explore.html")

@app.route("/communities")
def communities_page():
    return render_template("communities.html")

@app.route("/memes")
def memes_page():
    return render_template("memes.html")

@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


# ===================== STATIC FILES =====================

@app.route("/static/memes/<path:filename>")
def serve_meme_image(filename):
    """Serve generated meme images."""
    meme_dir = os.path.join(app.static_folder, "memes")
    return send_from_directory(meme_dir, filename)


# ===================== SSE ENDPOINT =====================

@app.route("/api/stream")
def sse_stream():
    """Server-Sent Events endpoint for real-time updates."""
    import queue
    q = queue.Queue(maxsize=50)
    sse_subscribers.append(q)
    
    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield data
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            if q in sse_subscribers:
                sse_subscribers.remove(q)
    
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ===================== API ENDPOINTS =====================

@app.route("/api/create", methods=["POST"])
def api_create():
    global sim
    with sim_lock:
        data = request.json or {}
        sim = Simulation(
            n_agents=int(data.get("n_agents", 50)),
            n_topics=int(data.get("n_topics", 20)),
            echo_chamber_strength=float(data.get("echo_chamber_strength", 0.5)),
            virality_sensitivity=float(data.get("virality_sensitivity", 1.0)),
            novelty_decay=float(data.get("novelty_decay", 0.95)),
            mutation_rate=float(data.get("mutation_rate", 0.3)),
            max_posts=500,
            community_detection_interval=5,
            bot_fraction=float(data.get("bot_fraction", 0.1)),
            generate_images=bool(data.get("generate_images", True)),
        )
    result = {
        "status": "ok", "tick": 0, 
        "n_agents": len(sim.agents),
        "n_bots": sum(1 for a in sim.agents if a.is_bot),
        "n_posts": len(sim.posts)
    }
    sse_notify("created", result)
    return json_response(result)


@app.route("/api/tick", methods=["POST"])
def api_tick():
    s = get_sim()
    if s is None:
        return json_response({"error": "No simulation"}, 400)
    with sim_lock:
        result = s.run_tick()
    sse_notify("tick", {"tick": s.tick, "n_posts": len(s.posts), "metrics": result})
    return json_response(result)


@app.route("/api/ticks", methods=["POST"])
def api_ticks():
    s = get_sim()
    if s is None:
        return json_response({"error": "No simulation"}, 400)
    n = int((request.json or {}).get("n", 10))
    with sim_lock:
        for _ in range(n):
            s.run_tick()
    return json_response({"status": "ok", "tick": s.tick})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global sim
    with sim_lock:
        sim = None
    return api_create()


@app.route("/api/auto-tick/start", methods=["POST"])
def api_auto_start():
    global auto_tick_running, auto_tick_thread, auto_tick_speed
    data = request.json or {}
    speed = float(data.get("speed", 1.0))
    auto_tick_speed = max(0.2, min(speed, 10.0))
    
    if not auto_tick_running:
        auto_tick_running = True
        auto_tick_thread = threading.Thread(target=auto_tick_loop, daemon=True)
        auto_tick_thread.start()
    
    return json_response({"status": "running", "speed": auto_tick_speed})


@app.route("/api/auto-tick/stop", methods=["POST"])
def api_auto_stop():
    global auto_tick_running
    auto_tick_running = False
    return json_response({"status": "stopped"})


@app.route("/api/auto-tick/status")
def api_auto_status():
    return json_response({
        "running": auto_tick_running,
        "speed": auto_tick_speed
    })


@app.route("/api/state")
def api_state():
    s = get_sim()
    if s is None:
        return json_response({"status": "none"})
    return json_response(s.get_state())


@app.route("/api/feed")
def api_feed():
    s = get_sim()
    if s is None:
        return json_response({"posts": []})
    sort_by = request.args.get("sort", "attention")
    limit = int(request.args.get("limit", 50))

    if sort_by == "recent":
        posts = sorted(s.posts, key=lambda p: p.tick_created, reverse=True)[:limit]
    elif sort_by == "engagement":
        posts = sorted(s.posts, key=lambda p: p.total_engagement, reverse=True)[:limit]
    elif sort_by == "controversial":
        posts = sorted(s.posts, key=lambda p: getattr(p, 'controversy_score', 0), reverse=True)[:limit]
    elif sort_by == "liked":
        posts = sorted(s.posts, key=lambda p: p.likes, reverse=True)[:limit]
    else:
        posts = sorted(s.posts, key=lambda p: p.attention_score, reverse=True)[:limit]

    result = []
    for p in posts:
        author = s.agent_map.get(p.author_id)
        result.append({
            "id": p.id,
            "author_name": author.name if author else "Unknown",
            "author_personality": author.personality.value if author else "",
            "author_followers": len(author.followers) if author else 0,
            "author_is_bot": getattr(author, 'is_bot', False) if author else False,
            "text": p.text,
            "content_type": p.content_type.value,
            "is_meme": isinstance(p, Meme),
            "image_path": getattr(p, 'image_path', None),
            "tick_created": p.tick_created,
            "views": p.views,
            "likes": p.likes,
            "reposts": p.reposts,
            "replies": p.replies,
            "quotes": getattr(p, 'quotes', 0),
            "engagement_rate": round(p.engagement_rate, 3),
            "attention_score": round(p.attention_score, 3),
            "novelty_score": round(p.novelty_score, 3),
            "generation": p.generation,
            "mutation_count": len(p.mutation_history),
            "controversy_score": round(getattr(p, 'controversy_score', 0), 3),
            "ratio_score": round(getattr(p, 'ratio_score', 0), 2),
            "distortion_level": round(getattr(p, 'distortion_level', 0), 3),
            "hashtags": getattr(p, 'hashtags', []),
            "is_bot_post": getattr(p, 'is_bot_post', False),
            "parent_id": p.parent_id,
            "thread_id": getattr(p, 'thread_id', None),
        })
    return json_response({"posts": result, "tick": s.tick, "total": len(s.posts)})


@app.route("/api/trending")
def api_trending():
    s = get_sim()
    if s is None:
        return json_response({"trending": [], "hashtags": []})

    top = sorted(s.posts, key=lambda p: p.attention_score, reverse=True)[:10]
    trending = []
    for p in top:
        author = s.agent_map.get(p.author_id)
        trending.append({
            "text": p.text[:80],
            "attention": round(p.attention_score, 2),
            "posts": p.reposts + p.replies,
            "likes": p.likes,
            "controversy": round(getattr(p, 'controversy_score', 0), 2),
            "author": author.name if author else "?",
            "hashtags": getattr(p, 'hashtags', []),
        })

    topic_totals = np.zeros(s.n_topics)
    for p in s.posts:
        topic_totals += p.topic_vector * p.attention_score

    hashtags = []
    for i, label in enumerate(TOPIC_LABELS[:s.n_topics]):
        if topic_totals[i] > 0.01:
            hashtags.append({"tag": f"#{label}", "volume": round(float(topic_totals[i]), 2)})
    hashtags.sort(key=lambda x: x["volume"], reverse=True)

    # Collect actual post hashtags
    post_hashtags = {}
    for p in s.posts:
        for tag in getattr(p, 'hashtags', []):
            post_hashtags[tag] = post_hashtags.get(tag, 0) + p.attention_score
    
    popular_hashtags = [{"tag": tag, "volume": round(vol, 2)} 
                        for tag, vol in sorted(post_hashtags.items(), key=lambda x: -x[1])[:15]]

    return json_response({
        "trending": trending, 
        "hashtags": hashtags[:15],
        "popular_hashtags": popular_hashtags,
    })


@app.route("/api/agents")
def api_agents():
    s = get_sim()
    if s is None:
        return json_response({"agents": []})
    sort_by = request.args.get("sort", "followers")
    agents_list = list(s.agents)
    if sort_by == "followers":
        agents_list.sort(key=lambda a: len(a.followers), reverse=True)
    elif sort_by == "posts":
        agents_list.sort(key=lambda a: a.posts_created, reverse=True)
    elif sort_by == "likes":
        agents_list.sort(key=lambda a: getattr(a, 'posts_liked', 0), reverse=True)
    elif sort_by == "views":
        agents_list.sort(key=lambda a: a.total_views, reverse=True)

    limit = int(request.args.get("limit", 30))
    result = [{
        "id": a.id, "name": a.name, "personality": a.personality.value,
        "followers": len(a.followers), "following": len(a.following),
        "posts_created": a.posts_created, "posts_liked": getattr(a, 'posts_liked', 0),
        "posts_replied": getattr(a, 'posts_replied', 0),
        "total_views": a.total_views,
        "total_likes_received": getattr(a, 'total_likes_received', 0),
        "emotional_state": round(a.emotional_state, 3), "community_id": a.community_id,
        "is_bot": getattr(a, 'is_bot', False),
        "bot_type": getattr(a, 'bot_type', None),
        "doom_scroll_depth": getattr(a, 'doom_scroll_depth', 0),
        "activity_level": round(a.get_activity_level(s.tick), 2),
    } for a in agents_list[:limit]]
    return json_response({"agents": result})


@app.route("/api/communities")
def api_communities():
    s = get_sim()
    if s is None:
        return json_response({"communities": []})

    result = []
    for cid, community in s.community_detector.communities.items():
        members = []
        for mid in list(community.members)[:20]:
            agent = s.agent_map.get(mid)
            if agent:
                members.append({
                    "name": agent.name, "personality": agent.personality.value, 
                    "followers": len(agent.followers),
                    "is_bot": getattr(agent, 'is_bot', False),
                })

        dialect = s.language_engine.community_dialects.get(cid, {})
        slang = list(s.language_engine.adopted_slang.get(cid, set()))[:10]
        top_words = s.language_engine.community_vocabularies.get(cid, {})
        top_words = top_words.most_common(10) if top_words else []

        result.append({
            "id": cid, "name": community.name, "size": len(community.members),
            "polarization": round(community.polarization_score, 3),
            "members": members, "dialect": dict(list(dialect.items())[:8]),
            "slang": slang, "top_words": top_words,
        })

    eco_pol = s.community_detector.compute_ecosystem_polarization()
    return json_response({"communities": result, "ecosystem_polarization": round(eco_pol, 3)})


@app.route("/api/memes")
def api_memes():
    s = get_sim()
    if s is None:
        return json_response({"memes": [], "families": [], "stats": {}})

    memes_list = sorted([p for p in s.posts if isinstance(p, Meme)], key=lambda p: p.attention_score, reverse=True)
    limit = int(request.args.get("limit", 30))
    result = [{
        "id": m.id, "text": m.text,
        "author": (s.agent_map.get(m.author_id).name if s.agent_map.get(m.author_id) else "?"),
        "generation": m.generation, "mutations": len(m.mutation_history),
        "attention": round(m.attention_score, 3), "views": m.views, "reposts": m.reposts,
        "likes": m.likes,
        "controversy": round(getattr(m, 'controversy_score', 0), 3),
        "distortion": round(getattr(m, 'distortion_level', 0), 3),
        "parent_meme_id": m.parent_meme_id,
        "mutation_types": [h["type"] for h in m.mutation_history],
        "hashtags": getattr(m, 'hashtags', []),
        "image_path": getattr(m, 'image_path', None),
    } for m in memes_list[:limit]]

    families = [{"root_id": rid, "size": len(desc), "descendants": desc[:10]}
                for rid, desc in s.meme_evolver.meme_families.items()]

    return json_response({"memes": result, "families": families, "stats": s.meme_evolver.to_dict()})


@app.route("/api/analytics")
def api_analytics():
    s = get_sim()
    if s is None or not s.metrics.tick_data:
        return json_response({"latest": {}, "history": {}})

    latest = s.metrics.tick_data[-1]
    history = {
        "ticks": [d["tick"] for d in s.metrics.tick_data],
        "mean_attention": [d.get("mean_attention", 0) for d in s.metrics.tick_data],
        "max_attention": [d.get("max_attention", 0) for d in s.metrics.tick_data],
        "attention_gini": [d.get("attention_gini", 0) for d in s.metrics.tick_data],
        "num_posts": [d.get("num_posts", 0) for d in s.metrics.tick_data],
        "num_communities": [d.get("num_communities", 0) for d in s.metrics.tick_data],
        "ecosystem_polarization": [d.get("ecosystem_polarization", 0) for d in s.metrics.tick_data],
        "language_divergence": [d.get("language_divergence", 0) for d in s.metrics.tick_data],
        "total_views": [d.get("total_views", 0) for d in s.metrics.tick_data],
        "total_likes": [d.get("total_likes", 0) for d in s.metrics.tick_data],
        "follower_gini": [d.get("follower_gini", 0) for d in s.metrics.tick_data],
        "influencer_dominance": [d.get("influencer_dominance", 0) for d in s.metrics.tick_data],
        "mean_controversy": [d.get("mean_controversy", 0) for d in s.metrics.tick_data],
        "mean_distortion": [d.get("mean_distortion", 0) for d in s.metrics.tick_data],
        "bot_post_fraction": [d.get("bot_post_fraction", 0) for d in s.metrics.tick_data],
        "ratioed_count": [d.get("ratioed_count", 0) for d in s.metrics.tick_data],
    }
    return json_response({
        "latest": latest, "history": history,
        "personality_distribution": latest.get("personality_distribution", {}),
        "total_posts_created": s.metrics.total_posts_created,
        "total_reposts": s.metrics.total_reposts,
        "total_likes": s.metrics.total_likes,
        "total_memes": s.metrics.total_memes_created,
        "total_mutations": s.metrics.total_mutations,
        "sentiment_wave": s.active_sentiment_wave,
        "follow_cascades_count": len(s.follow_cascades),
        "neural_net": s.neural_ranker.get_status(),
    })


@app.route("/api/language")
def api_language():
    s = get_sim()
    if s is None:
        return json_response({"divergence": 0, "global_vocab": [], "dialects": {}})

    le = s.language_engine
    top_words = le.global_vocabulary.most_common(20)
    dialects = {str(cid): dict(list(d.items())[:10]) for cid, d in le.community_dialects.items()}

    return json_response({
        "divergence": round(le.divergence_history[-1], 3) if le.divergence_history else 0,
        "divergence_history": le.divergence_history[-50:],
        "global_vocab_size": len(le.global_vocabulary),
        "top_words": top_words, "dialects": dialects,
        "adopted_slang": {str(k): list(v) for k, v in le.adopted_slang.items()},
    })


@app.route("/api/platform")
def api_platform():
    s = get_sim()
    if s is None:
        return json_response({})
    return json_response(s.platform.to_dict())


@app.route("/api/neural-net")
def api_neural_net():
    s = get_sim()
    if s is None:
        return json_response({"status": "no_simulation"})
    return json_response(s.neural_ranker.get_status())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
