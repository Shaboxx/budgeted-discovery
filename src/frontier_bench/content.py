"""Explicit synthetic emissions, separate from topology, relevance, and adoption."""
from __future__ import annotations

import numpy as np

from .schemas import digest

NICHE_VOCABULARY = ("tabletop", "dice", "campaign", "miniatures", "quest", "rulebook")
OTHER_VOCABULARY = ("garden", "recipe", "workout", "travel", "painting", "camera")
SHARED_VOCABULARY = ("community", "story", "guide", "design", "event", "update")
TOPIC_VOCABULARY = ("lantern", "festival", "preview", "release")


def emit_content(accounts: list[dict], snapshots: list, states: list[dict],
                 relevant_accounts: set[str], config: dict, seed: int) -> tuple[list[dict], dict]:
    """Generate ordinary lexical content using fixed relevance and transient state.

    Background emissions are Bernoulli per account/tick. Adoption adds an
    independent Bernoulli topic post while status=1. Stationary control freezes
    the adoption input at tick zero, retaining stochastic equal-rate emissions.
    Fixture mode emits exactly fixture_posts_per_tick per account/tick. Optional
    declared activity change multiplies emissions after change_tick.
    """
    rng = np.random.default_rng(seed)
    background = float(config.get("background_probability", 0.25))
    topic = float(config.get("topic_probability", 0.8))
    ambiguity = float(config.get("ambiguity_probability", 0.15))
    hashtag_probability = float(config.get("hashtag_probability", 0.6))
    mention_probability = float(config.get("mention_probability", 0.1))
    for value in (background, topic, ambiguity, hashtag_probability, mention_probability):
        if not 0 <= value <= 1:
            raise ValueError("Content probabilities must lie in [0, 1]")
    delay = int(config.get("index_delay", 1))
    delay_jitter = int(config.get("index_delay_jitter", 0))
    if delay < 0 or delay_jitter < 0:
        raise ValueError("Indexing delay cannot be negative")
    fixture = config.get("fixture", "") == "stationary"
    fixture_count = int(config.get("fixture_posts_per_tick", 1))
    if fixture_count < 0:
        raise ValueError("Fixture post count cannot be negative")
    stationary = bool(config.get("stationary", False)) or fixture
    change_tick = config.get("change_tick")
    multiplier = int(config.get("change_multiplier", 2))
    if multiplier < 1:
        raise ValueError("Activity multiplier must be a positive integer")
    nodes = sorted(account["id"] for account in accounts)
    posts = []
    for tick, edges in enumerate(snapshots):
        neighbors = {node: set() for node in nodes}
        for source, target in edges:
            neighbors[source].add(target)
            neighbors[target].add(source)
        factor = multiplier if change_tick is not None and tick >= int(change_tick) else 1
        for account in nodes:
            active = states[0 if stationary else tick][account] == 1
            if fixture:
                kinds = [False] * (fixture_count * factor)
            else:
                kinds = []
                for _ in range(factor):
                    if rng.random() < background:
                        kinds.append(False)
                    if active and rng.random() < topic:
                        kinds.append(True)
            for sequence, is_topic in enumerate(kinds):
                is_niche = account in relevant_accounts
                lexical_niche = is_niche if fixture else (is_niche != (rng.random() < ambiguity))
                vocabulary = NICHE_VOCABULARY if lexical_niche else OTHER_VOCABULARY
                if fixture:
                    terms = [vocabulary[0], vocabulary[1], SHARED_VOCABULARY[0]]
                    hashtags, mentions = [vocabulary[0]], []
                else:
                    terms = rng.choice(vocabulary, size=2, replace=False).tolist()
                    terms += [str(rng.choice(SHARED_VOCABULARY))]
                    if is_topic:
                        terms += rng.choice(TOPIC_VOCABULARY, size=2, replace=False).tolist()
                    hashtags = [terms[0]] if rng.random() < hashtag_probability else []
                    contacts = sorted(neighbors[account])
                    mentions = ([str(rng.choice(contacts))]
                                if contacts and rng.random() < mention_probability else [])
                available_at = tick + delay + (int(rng.integers(delay_jitter + 1)) if delay_jitter else 0)
                text = " ".join(terms) + "".join(" #" + tag for tag in hashtags)
                posts.append({
                    "id": "p:" + digest([seed, account, tick, sequence])[:24],
                    "author": account, "event_at": tick, "available_at": available_at,
                    "text": text, "terms": terms, "hashtags": hashtags,
                    "mentions": mentions,
                })
    posts.sort(key=lambda post: (post["event_at"], post["id"]))
    return posts, {
        "version": "independent-bernoulli-emission-v1", "seed": int(seed),
        "background_probability": background, "topic_probability": topic,
        "ambiguity_probability": ambiguity, "fixture": "stationary" if fixture else None,
        "stationary_adoption_input": stationary, "index_delay": delay,
        "index_delay_jitter": delay_jitter, "change_tick": change_tick,
        "change_multiplier": multiplier, "posts": len(posts),
        "posts_per_tick": [sum(p["event_at"] == t for p in posts) for t in range(len(snapshots))],
        "event_order": "diffusion state at tick; emission; indexing delay; policy acquisition",
        "enduring_relevance": "fixed before policy execution; noisy ordinary lexical emission",
    }
