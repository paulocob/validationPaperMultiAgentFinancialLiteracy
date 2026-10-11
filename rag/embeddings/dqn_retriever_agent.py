"""
DQN (Deep Q-Network) Agent for Adaptive Retriever and Reranking Selection.

Core contribution of the doctoral thesis: connects the RAG pipeline with
Deep Reinforcement Learning. The agent learns to select the optimal
(retriever, alpha) combination for each query based on query features.

State Space (~17 dims):
  - query_length (1): number of characters (normalized)
  - country (3): one-hot [BR, PT, AO]
  - topic (6): one-hot [budget, savings, debt, investment, credit, cultural]
  - difficulty (3): one-hot [basic, intermediate, advanced]
  - has_audio (1): binary flag
  - query_embedding_stats (2): mean and std of query embedding
  - has_cultural_notes (1): binary flag

Action Space (6 discrete actions):
  0: LoRA+CL, alpha=1.0 (no reranking)
  1: LoRA+CL, alpha=0.9
  2: LoRA+CL, alpha=0.8
  3: e5-multilingual, alpha=1.0
  4: e5-multilingual, alpha=0.9
  5: e5-multilingual, alpha=0.8

Reward: MRR of the retrieval + equity bonus (reducing inter-country variance).

Training protocol:
  1. Pre-compute all retrieval results for 999 queries x 6 actions
  2. Train for 100 episodes (full pass through all queries per episode)
  3. Epsilon-greedy exploration with decay

Usage:
    python3 -m embeddings.dqn_retriever_agent
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

console = Console()

# Caminho relativo ao projeto: o original apontava para uma pasta local do autor.
FIGURES_DIR = _PROJECT_ROOT / "embeddings" / "results" / "figures"
OUTPUT_DIR = _PROJECT_ROOT / "embeddings" / "results"

# Action space definition
ACTIONS = [
    {"retriever": "LoRA+CL",          "alpha": 1.0},
    {"retriever": "LoRA+CL",          "alpha": 0.9},
    {"retriever": "LoRA+CL",          "alpha": 0.8},
    {"retriever": "e5-multilingual",   "alpha": 1.0},
    {"retriever": "e5-multilingual",   "alpha": 0.9},
    {"retriever": "e5-multilingual",   "alpha": 0.8},
]
NUM_ACTIONS = len(ACTIONS)

# Feature encoding mappings
COUNTRIES = ["BR", "PT", "AO"]
TOPICS = ["budget", "credit", "cultural", "debt", "investment", "savings"]
DIFFICULTIES = ["advanced", "basic", "intermediate"]

# DQN hyperparameters
STATE_DIM = 17  # 1 + 3 + 6 + 3 + 1 + 2 + 1
HIDDEN1 = 64
HIDDEN2 = 32
LR = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_EPISODES = 80
REPLAY_BUFFER_SIZE = 2000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 50
NUM_EPISODES = 100
EQUITY_BONUS_WEIGHT = 0.05
EQUITY_WINDOW_SIZE = 50

# Retrieval parameters
TOP_K_RETRIEVE = 20
K_VALUES = [1, 3, 5, 10]


# =========================================================================
# Model loaders (reused from rag_reranker_experiment.py)
# =========================================================================

def _get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_openclip_lora_cl():
    import open_clip
    from embeddings.clip_lora_finetune import apply_lora_simple, load_lora_weights

    lora_path = _PROJECT_ROOT / "embeddings" / "lora_output_cl" / "best_lora.pt"
    if not lora_path.exists():
        console.print(f"  [yellow]LoRA weights not found at {lora_path}[/yellow]")
        return None, "LoRA+CL (not found)"

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.float()

    config_path = lora_path.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    apply_lora_simple(model, rank=config.get("rank", 4), alpha=config.get("alpha", 8.0))
    load_lora_weights(model, lora_path)

    device = _get_device()
    model = model.to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        tokens = tokenizer(texts)[:, :77].to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()
    return encode, "LoRA+CL"


def load_e5_multilingual():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    def encode(texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return model.encode(prefixed, normalize_embeddings=True)
    return encode, "e5-multilingual"


def encode_batch(texts: list[str], encode_fn, batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encode_fn(batch)
        all_embs.append(emb)
    return np.vstack(all_embs)


# =========================================================================
# Corpus and context builders
# =========================================================================

def build_corpus_base(qa_pairs: list[QAPair]) -> list[dict]:
    docs = []
    for qa in qa_pairs:
        docs.append({
            "id": f"{qa.id}_answer", "qa_id": qa.id, "text": qa.answer,
            "type": "answer", "country": qa.country, "topic": qa.topic,
        })
        if qa.image_description:
            docs.append({
                "id": f"{qa.id}_image", "qa_id": qa.id, "text": qa.image_description,
                "type": "image_desc", "country": qa.country, "topic": qa.topic,
            })
    return docs


def build_context_maps(qa_pairs: list[QAPair]) -> tuple[dict, dict]:
    cultural_map, audio_map = {}, {}
    for qa in qa_pairs:
        if qa.cultural_notes:
            cultural_map[qa.id] = qa.cultural_notes
        if qa.audio_transcript:
            audio_map[qa.id] = qa.audio_transcript
    return cultural_map, audio_map


# =========================================================================
# Pre-compute all retrieval results for each (retriever, alpha) action
# =========================================================================

def precompute_retrievals_for_model(
    qa_pairs: list[QAPair],
    corpus: list[dict],
    encode_fn,
    cultural_map: dict[str, str],
    audio_map: dict[str, str],
    alphas: list[float],
    top_k_retrieve: int = TOP_K_RETRIEVE,
    batch_size: int = 64,
) -> dict[float, dict]:
    """
    Pre-compute retrieval results for one retriever across multiple alpha values.

    Returns dict: alpha -> {
        "mrr": np.array of per-query MRR,
        "hit@k": np.array of per-query Hit@k for each k,
        "query_embs": query embeddings (for state features),
    }
    """
    # Encode corpus
    console.print("    Encoding corpus...")
    corpus_texts = [d["text"] for d in corpus]
    corpus_embs = encode_batch(corpus_texts, encode_fn, batch_size)

    # Build qa_id -> corpus indices
    qa_to_corpus: dict[str, list[int]] = {}
    for idx, doc in enumerate(corpus):
        qa_to_corpus.setdefault(doc["qa_id"], []).append(idx)

    # Encode queries
    console.print("    Encoding queries...")
    queries = [qa.question for qa in qa_pairs]
    query_embs = encode_batch(queries, encode_fn, batch_size)

    # Encode cultural notes
    console.print("    Encoding cultural notes...")
    cultural_embs = {}
    cultural_ids = list(cultural_map.keys())
    if cultural_ids:
        cultural_texts = [cultural_map[qid] for qid in cultural_ids]
        embs = encode_batch(cultural_texts, encode_fn, batch_size)
        for j, qid in enumerate(cultural_ids):
            cultural_embs[qid] = embs[j]

    # Encode audio transcripts
    console.print("    Encoding audio transcripts...")
    audio_embs = {}
    audio_ids = list(audio_map.keys())
    if audio_ids:
        audio_texts = [audio_map[qid] for qid in audio_ids]
        embs = encode_batch(audio_texts, encode_fn, batch_size)
        for j, qid in enumerate(audio_ids):
            audio_embs[qid] = embs[j]

    # Pre-compute base similarities for all queries
    console.print("    Computing similarities...")
    # query_embs: (n_queries, d), corpus_embs: (n_docs, d)
    all_sims = query_embs @ corpus_embs.T  # (n_queries, n_docs)

    results = {}
    max_k = max(K_VALUES)

    for alpha in alphas:
        console.print(f"    Evaluating alpha={alpha}...")
        mrr_arr = np.zeros(len(qa_pairs))
        hit_arrs = {k: np.zeros(len(qa_pairs)) for k in K_VALUES}

        for i, qa in enumerate(qa_pairs):
            sims = all_sims[i]
            q_emb = query_embs[i]

            # Step 1: retrieve top-K
            top_indices = np.argsort(sims)[::-1][:top_k_retrieve]

            # Step 2: rerank
            if alpha < 1.0:
                reranked_scores = []
                for idx in top_indices:
                    original_sim = sims[idx]
                    doc_qa_id = corpus[idx]["qa_id"]

                    cultural_sim = 0.0
                    if doc_qa_id in cultural_embs:
                        cultural_sim = float(q_emb @ cultural_embs[doc_qa_id])

                    audio_sim = 0.0
                    if doc_qa_id in audio_embs:
                        audio_sim = float(q_emb @ audio_embs[doc_qa_id])

                    context_sim = max(cultural_sim, audio_sim)
                    reranked = alpha * original_sim + (1.0 - alpha) * context_sim
                    reranked_scores.append(reranked)

                reranked_order = np.argsort(reranked_scores)[::-1]
                top_indices = top_indices[reranked_order]

            # Ground truth
            gt_indices = set(qa_to_corpus.get(qa.id, []))

            # MRR
            rr = 0.0
            for rank, idx in enumerate(top_indices[:max_k]):
                if idx in gt_indices:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_arr[i] = rr

            # Hit@k
            for k in K_VALUES:
                retrieved = set(top_indices[:k].tolist())
                hit_arrs[k][i] = 1.0 if len(retrieved & gt_indices) > 0 else 0.0

        results[alpha] = {
            "mrr": mrr_arr,
            "hit": hit_arrs,
            "query_embs": query_embs,
        }
        console.print(f"      MRR={mrr_arr.mean():.4f}  Hit@5={hit_arrs[5].mean():.3f}")

    return results


# =========================================================================
# State feature extraction
# =========================================================================

def extract_state(qa: QAPair, query_emb: np.ndarray) -> np.ndarray:
    """Extract 17-dim state vector from a QA pair and its query embedding."""
    features = []

    # 1. query_length (normalized by max observed ~500 chars)
    features.append(min(len(qa.question) / 500.0, 2.0))

    # 2. country one-hot (3 dims)
    for c in COUNTRIES:
        features.append(1.0 if qa.country == c else 0.0)

    # 3. topic one-hot (6 dims)
    for t in TOPICS:
        features.append(1.0 if qa.topic == t else 0.0)

    # 4. difficulty one-hot (3 dims)
    for d in DIFFICULTIES:
        features.append(1.0 if qa.difficulty == d else 0.0)

    # 5. has_audio (1 dim)
    features.append(1.0 if qa.audio_transcript else 0.0)

    # 6. query_embedding_stats: mean and std (2 dims)
    features.append(float(np.mean(query_emb)))
    features.append(float(np.std(query_emb)))

    # 7. has_cultural_notes (1 dim)
    features.append(1.0 if qa.cultural_notes else 0.0)

    return np.array(features, dtype=np.float32)


# =========================================================================
# DQN Network
# =========================================================================

class DQNetwork(nn.Module):
    """Simple feedforward DQN: state_dim -> 64 -> 32 -> num_actions."""

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================================
# Experience Replay Buffer
# =========================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================================
# DQN Agent
# =========================================================================

class DQNAgent:
    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS):
        self.device = "cpu"  # Small network, CPU is fine
        self.num_actions = num_actions

        self.q_network = DQNetwork(state_dim, num_actions).to(self.device)
        self.target_network = DQNetwork(state_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.epsilon = EPSILON_START
        self.step_count = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def update(self) -> float | None:
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0]
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self, episode: int):
        self.epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES
        )


# =========================================================================
# Training
# =========================================================================

def train_dqn(
    qa_pairs: list[QAPair],
    precomputed: dict[str, dict[float, dict]],
    num_episodes: int = NUM_EPISODES,
) -> tuple[DQNAgent, dict]:
    """
    Train the DQN agent using pre-computed retrieval results.

    precomputed: { "LoRA+CL": {alpha: {"mrr": arr, "hit": {k: arr}, "query_embs": arr}},
                   "e5-multilingual": {alpha: {...}} }
    """
    agent = DQNAgent()
    n_queries = len(qa_pairs)

    # Build state vectors for each query using embeddings from LoRA+CL (primary)
    lora_query_embs = precomputed["LoRA+CL"][1.0]["query_embs"]
    states = np.array([
        extract_state(qa, lora_query_embs[i])
        for i, qa in enumerate(qa_pairs)
    ])

    # Build reward matrix: (n_queries, n_actions)
    reward_matrix = np.zeros((n_queries, NUM_ACTIONS))
    for action_idx, action in enumerate(ACTIONS):
        retriever = action["retriever"]
        alpha = action["alpha"]
        reward_matrix[:, action_idx] = precomputed[retriever][alpha]["mrr"]

    # Build hit@5 matrix for evaluation
    hit5_matrix = np.zeros((n_queries, NUM_ACTIONS))
    for action_idx, action in enumerate(ACTIONS):
        retriever = action["retriever"]
        alpha = action["alpha"]
        hit5_matrix[:, action_idx] = precomputed[retriever][alpha]["hit"][5]

    # Country indices for equity bonus
    country_indices = {c: [] for c in COUNTRIES}
    for i, qa in enumerate(qa_pairs):
        country_indices[qa.country].append(i)

    # Training loop
    training_history = {
        "episode": [], "avg_reward": [], "avg_mrr": [], "avg_hit5": [],
        "epsilon": [], "loss": [],
        "mrr_br": [], "mrr_pt": [], "mrr_ao": [],
        "action_counts": [],
    }

    console.print(f"\n[bold cyan]Starting DQN training: {num_episodes} episodes, {n_queries} queries/episode[/bold cyan]")

    # Sliding window for equity computation
    country_reward_window = {c: deque(maxlen=EQUITY_WINDOW_SIZE) for c in COUNTRIES}

    for episode in range(num_episodes):
        t0 = time.time()
        episode_rewards = []
        episode_losses = []
        action_counts = np.zeros(NUM_ACTIONS, dtype=int)
        query_order = list(range(n_queries))
        random.shuffle(query_order)

        per_country_mrr = {c: [] for c in COUNTRIES}

        for step, qi in enumerate(query_order):
            state = states[qi]
            action = agent.select_action(state)
            action_counts[action] += 1

            # Base reward: MRR for this (query, action)
            base_reward = reward_matrix[qi, action]

            # Equity bonus: reward actions that reduce inter-country variance
            country = qa_pairs[qi].country
            country_reward_window[country].append(base_reward)

            equity_bonus = 0.0
            if all(len(country_reward_window[c]) >= 10 for c in COUNTRIES):
                country_means = [np.mean(list(country_reward_window[c])) for c in COUNTRIES]
                variance = np.var(country_means)
                # Lower variance = higher bonus (normalized to ~[0, 0.1])
                equity_bonus = EQUITY_BONUS_WEIGHT * max(0, 1.0 - variance * 100)

            reward = base_reward + equity_bonus

            # Next state (next query in shuffled order, or same if last)
            next_qi = query_order[(step + 1) % n_queries]
            next_state = states[next_qi]
            done = (step == n_queries - 1)

            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # Update Q-network
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            episode_rewards.append(base_reward)
            per_country_mrr[country].append(base_reward)

        # Decay epsilon
        agent.decay_epsilon(episode)

        # Episode metrics
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        # Compute greedy policy metrics
        greedy_actions = np.array([agent.select_action(states[i], greedy=True) for i in range(n_queries)])
        greedy_mrrs = np.array([reward_matrix[i, greedy_actions[i]] for i in range(n_queries)])
        greedy_hit5 = np.array([hit5_matrix[i, greedy_actions[i]] for i in range(n_queries)])

        mrr_by_country = {}
        for c in COUNTRIES:
            idx = country_indices[c]
            mrr_by_country[c] = greedy_mrrs[idx].mean()

        elapsed = time.time() - t0

        training_history["episode"].append(episode)
        training_history["avg_reward"].append(avg_reward)
        training_history["avg_mrr"].append(greedy_mrrs.mean())
        training_history["avg_hit5"].append(greedy_hit5.mean())
        training_history["epsilon"].append(agent.epsilon)
        training_history["loss"].append(avg_loss)
        training_history["mrr_br"].append(mrr_by_country["BR"])
        training_history["mrr_pt"].append(mrr_by_country["PT"])
        training_history["mrr_ao"].append(mrr_by_country["AO"])
        training_history["action_counts"].append(action_counts.copy())

        if episode % 10 == 0 or episode == num_episodes - 1:
            console.print(
                f"  Ep {episode:3d}/{num_episodes} | "
                f"MRR={greedy_mrrs.mean():.4f} Hit@5={greedy_hit5.mean():.3f} | "
                f"eps={agent.epsilon:.3f} loss={avg_loss:.5f} | "
                f"BR={mrr_by_country['BR']:.4f} PT={mrr_by_country['PT']:.4f} AO={mrr_by_country['AO']:.4f} | "
                f"{elapsed:.1f}s"
            )

    return agent, training_history


# =========================================================================
# Evaluation
# =========================================================================

def evaluate_baselines_and_dqn(
    qa_pairs: list[QAPair],
    precomputed: dict[str, dict[float, dict]],
    agent: DQNAgent,
    lora_query_embs: np.ndarray,
) -> pd.DataFrame:
    """Compare DQN policy vs fixed baselines."""
    n = len(qa_pairs)
    states = np.array([extract_state(qa, lora_query_embs[i]) for i, qa in enumerate(qa_pairs)])

    # Build reward and hit matrices
    reward_matrix = np.zeros((n, NUM_ACTIONS))
    hit5_matrix = np.zeros((n, NUM_ACTIONS))
    for ai, action in enumerate(ACTIONS):
        r = action["retriever"]
        a = action["alpha"]
        reward_matrix[:, ai] = precomputed[r][a]["mrr"]
        hit5_matrix[:, ai] = precomputed[r][a]["hit"][5]

    # Country indices
    country_indices = {c: [] for c in COUNTRIES}
    for i, qa in enumerate(qa_pairs):
        country_indices[qa.country].append(i)

    # Define policies
    policies = {
        "LoRA+CL a=1.0 (baseline)":   np.full(n, 0, dtype=int),
        "LoRA+CL a=0.9 (best fixed)": np.full(n, 1, dtype=int),
        "e5 a=0.8":                    np.full(n, 5, dtype=int),
        "Random":                      np.random.randint(0, NUM_ACTIONS, size=n),
        "Oracle (best per query)":     reward_matrix.argmax(axis=1),
        "DQN Policy":                  np.array([agent.select_action(states[i], greedy=True) for i in range(n)]),
    }

    rows = []
    for policy_name, actions in policies.items():
        mrrs = np.array([reward_matrix[i, actions[i]] for i in range(n)])
        hit5s = np.array([hit5_matrix[i, actions[i]] for i in range(n)])

        # Per-country MRR
        mrr_country = {}
        for c in COUNTRIES:
            idx = country_indices[c]
            mrr_country[c] = mrrs[idx].mean()

        # Kruskal-Wallis test
        kw_groups = [mrrs[country_indices[c]] for c in COUNTRIES]
        h_stat, p_val = stats.kruskal(*kw_groups)

        variance = np.var(list(mrr_country.values()))
        spread = max(mrr_country.values()) - min(mrr_country.values())

        rows.append({
            "policy": policy_name,
            "avg_mrr": mrrs.mean(),
            "avg_hit5": hit5s.mean(),
            "mrr_br": mrr_country["BR"],
            "mrr_pt": mrr_country["PT"],
            "mrr_ao": mrr_country["AO"],
            "variance": variance,
            "spread": spread,
            "kw_h": h_stat,
            "kw_p": p_val,
        })

    return pd.DataFrame(rows)


def analyze_dqn_actions(
    qa_pairs: list[QAPair],
    agent: DQNAgent,
    lora_query_embs: np.ndarray,
) -> pd.DataFrame:
    """Analyze which actions the DQN selects for different query types."""
    n = len(qa_pairs)
    states = np.array([extract_state(qa, lora_query_embs[i]) for i, qa in enumerate(qa_pairs)])
    actions = np.array([agent.select_action(states[i], greedy=True) for i in range(n)])

    records = []
    for i, qa in enumerate(qa_pairs):
        a = actions[i]
        records.append({
            "qa_id": qa.id,
            "country": qa.country,
            "topic": qa.topic,
            "difficulty": qa.difficulty,
            "has_audio": 1 if qa.audio_transcript else 0,
            "action": a,
            "retriever": ACTIONS[a]["retriever"],
            "alpha": ACTIONS[a]["alpha"],
        })

    return pd.DataFrame(records)


# =========================================================================
# Visualization
# =========================================================================

def plot_training_curve(history: dict, output_dir: Path) -> Path:
    """Plot DQN training curves: reward, MRR, epsilon, per-country MRR."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    eps = history["episode"]

    # Plot 1: Greedy policy MRR over episodes
    ax = axes[0, 0]
    ax.plot(eps, history["avg_mrr"], linewidth=2, color="#1E88E5", label="DQN greedy MRR")
    # Add horizontal lines for fixed baselines (will be filled in main)
    ax.set_xlabel("Episode")
    ax.set_ylabel("MRR")
    ax.set_title("DQN Greedy Policy MRR During Training", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Epsilon decay and loss
    ax1 = axes[0, 1]
    color1 = "#E53935"
    ax1.plot(eps, history["epsilon"], linewidth=2, color=color1, label="Epsilon")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Epsilon", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title("Exploration (Epsilon) and Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#43A047"
    ax2.plot(eps, history["loss"], linewidth=1.5, color=color2, alpha=0.7, label="Loss")
    ax2.set_ylabel("Loss", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Plot 3: Per-country MRR over episodes
    ax = axes[1, 0]
    country_colors = {"BR": "#2196F3", "PT": "#4CAF50", "AO": "#FF9800"}
    for c in COUNTRIES:
        ax.plot(eps, history[f"mrr_{c.lower()}"], linewidth=2, color=country_colors[c], label=c)
    ax.set_xlabel("Episode")
    ax.set_ylabel("MRR")
    ax.set_title("Per-Country MRR (DQN Greedy Policy)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Action distribution over episodes (stacked area)
    ax = axes[1, 1]
    action_counts_arr = np.array(history["action_counts"])  # (n_episodes, 6)
    # Normalize to proportions
    action_props = action_counts_arr / action_counts_arr.sum(axis=1, keepdims=True)
    action_labels = [f"{ACTIONS[i]['retriever']} a={ACTIONS[i]['alpha']}" for i in range(NUM_ACTIONS)]
    action_colors = ["#E53935", "#FF7043", "#FFAB91", "#1E88E5", "#42A5F5", "#90CAF9"]

    ax.stackplot(eps, action_props.T, labels=action_labels, colors=action_colors, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Proportion")
    ax.set_title("Action Distribution Over Training", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.suptitle("FinEduBench-Luso: DQN Retriever Agent Training",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "dqn_training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_action_distribution(action_df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap of action distribution by country x topic and country x difficulty."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    action_labels = [f"{ACTIONS[i]['retriever']}\na={ACTIONS[i]['alpha']}" for i in range(NUM_ACTIONS)]

    # Heatmap 1: Country x Action
    ax = axes[0]
    ct = pd.crosstab(action_df["country"], action_df["action"], normalize="index")
    # Ensure all actions present
    for a in range(NUM_ACTIONS):
        if a not in ct.columns:
            ct[a] = 0.0
    ct = ct[sorted(ct.columns)]
    ct.columns = [action_labels[i] for i in ct.columns]
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Proportion"})
    ax.set_title("Action Distribution by Country", fontsize=11, fontweight="bold")
    ax.set_ylabel("Country")
    ax.set_xlabel("Action")

    # Heatmap 2: Topic x Action
    ax = axes[1]
    ct2 = pd.crosstab(action_df["topic"], action_df["action"], normalize="index")
    for a in range(NUM_ACTIONS):
        if a not in ct2.columns:
            ct2[a] = 0.0
    ct2 = ct2[sorted(ct2.columns)]
    ct2.columns = [action_labels[i] for i in ct2.columns]
    sns.heatmap(ct2, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Proportion"})
    ax.set_title("Action Distribution by Topic", fontsize=11, fontweight="bold")
    ax.set_ylabel("Topic")
    ax.set_xlabel("Action")

    # Heatmap 3: Difficulty x Action
    ax = axes[2]
    ct3 = pd.crosstab(action_df["difficulty"], action_df["action"], normalize="index")
    for a in range(NUM_ACTIONS):
        if a not in ct3.columns:
            ct3[a] = 0.0
    ct3 = ct3[sorted(ct3.columns)]
    ct3.columns = [action_labels[i] for i in ct3.columns]
    sns.heatmap(ct3, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Proportion"})
    ax.set_title("Action Distribution by Difficulty", fontsize=11, fontweight="bold")
    ax.set_ylabel("Difficulty")
    ax.set_xlabel("Action")

    plt.suptitle("FinEduBench-Luso: DQN Agent Action Distribution (Learned Policy)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "dqn_action_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart comparing DQN vs baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    policies = comparison_df["policy"].values
    n_policies = len(policies)

    # Color coding: highlight DQN and Oracle
    colors = []
    for p in policies:
        if "DQN" in p:
            colors.append("#E53935")
        elif "Oracle" in p:
            colors.append("#FFD600")
        elif "best fixed" in p:
            colors.append("#43A047")
        elif "Random" in p:
            colors.append("#9E9E9E")
        else:
            colors.append("#1E88E5")

    # Plot 1: MRR comparison
    ax = axes[0]
    bars = ax.bar(range(n_policies), comparison_df["avg_mrr"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, comparison_df["avg_mrr"]):
        ax.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(n_policies))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MRR")
    ax.set_title("Average MRR: DQN vs Baselines", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Per-country MRR (grouped)
    ax = axes[1]
    x = np.arange(n_policies)
    width = 0.25
    country_colors_map = {"BR": "#2196F3", "PT": "#4CAF50", "AO": "#FF9800"}
    for ci, c in enumerate(COUNTRIES):
        vals = comparison_df[f"mrr_{c.lower()}"]
        bars = ax.bar(x + (ci - 1) * width, vals, width, label=c,
                      color=country_colors_map[c], edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MRR")
    ax.set_title("Per-Country MRR", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Equity (KW p-value)
    ax = axes[2]
    bars = ax.bar(range(n_policies), comparison_df["kw_p"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, comparison_df["kw_p"]):
        ax.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="p=0.05")
    ax.set_xticks(range(n_policies))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Kruskal-Wallis p-value")
    ax.set_title("Inter-Country Equity (higher p = more equitable)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: DQN Retriever Agent vs Fixed Baselines",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "dqn_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# =========================================================================
# Main
# =========================================================================

def main():
    console.print("\n[bold magenta]{'='*20} DQN Retriever Agent {'='*20}[/bold magenta]")
    console.print("[bold]Deep Reinforcement Learning for Adaptive RAG Pipeline Selection[/bold]\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ---- Step 1: Load dataset ----
    console.print("[bold green]Step 1: Loading dataset...[/bold green]")
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs loaded.")

    corpus = build_corpus_base(qa_pairs)
    console.print(f"  Corpus: {len(corpus)} documents (answers + image_descriptions)")

    cultural_map, audio_map = build_context_maps(qa_pairs)
    console.print(f"  Cultural notes: {len(cultural_map)} | Audio transcripts: {len(audio_map)}")

    # ---- Step 2: Pre-compute all retrieval results ----
    console.print("\n[bold green]Step 2: Pre-computing retrieval results for all 6 actions...[/bold green]")
    alphas = [1.0, 0.9, 0.8]
    precomputed: dict[str, dict[float, dict]] = {}

    # LoRA+CL
    console.print("\n  [bold cyan]Loading LoRA+CL...[/bold cyan]")
    encode_fn_lora, _ = load_openclip_lora_cl()
    if encode_fn_lora is None:
        console.print("  [red]LoRA+CL model not found. Cannot proceed.[/red]")
        return

    t0 = time.time()
    precomputed["LoRA+CL"] = precompute_retrievals_for_model(
        qa_pairs, corpus, encode_fn_lora, cultural_map, audio_map, alphas
    )
    console.print(f"  LoRA+CL pre-computation done in {time.time()-t0:.1f}s")

    # Free LoRA+CL memory
    del encode_fn_lora
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # e5-multilingual
    console.print("\n  [bold cyan]Loading e5-multilingual...[/bold cyan]")
    encode_fn_e5, _ = load_e5_multilingual()

    t0 = time.time()
    precomputed["e5-multilingual"] = precompute_retrievals_for_model(
        qa_pairs, corpus, encode_fn_e5, cultural_map, audio_map, alphas
    )
    console.print(f"  e5-multilingual pre-computation done in {time.time()-t0:.1f}s")

    # Free e5 memory
    del encode_fn_e5
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Print pre-computed results summary
    console.print("\n[bold]Pre-computed retrieval summary:[/bold]")
    summary_table = Table(title="Pre-computed MRR for all (retriever, alpha) combinations", show_lines=True)
    summary_table.add_column("Action", style="bold")
    summary_table.add_column("Retriever")
    summary_table.add_column("Alpha", justify="right")
    summary_table.add_column("MRR", justify="right")
    summary_table.add_column("Hit@5", justify="right")
    for ai, action in enumerate(ACTIONS):
        r, a = action["retriever"], action["alpha"]
        mrr = precomputed[r][a]["mrr"].mean()
        hit5 = precomputed[r][a]["hit"][5].mean()
        summary_table.add_row(str(ai), r, f"{a:.1f}", f"{mrr:.4f}", f"{hit5:.3f}")
    console.print(summary_table)

    # ---- Step 3: Train DQN ----
    console.print("\n[bold green]Step 3: Training DQN agent...[/bold green]")
    agent, history = train_dqn(qa_pairs, precomputed, NUM_EPISODES)

    # ---- Step 4: Evaluate ----
    console.print("\n[bold green]Step 4: Evaluating DQN vs baselines...[/bold green]")
    lora_query_embs = precomputed["LoRA+CL"][1.0]["query_embs"]
    comparison_df = evaluate_baselines_and_dqn(qa_pairs, precomputed, agent, lora_query_embs)

    # Print comparison table
    comp_table = Table(title="DQN vs Baselines", show_lines=True)
    comp_table.add_column("Policy", style="bold")
    comp_table.add_column("MRR", justify="right")
    comp_table.add_column("Hit@5", justify="right")
    comp_table.add_column("MRR_BR", justify="right")
    comp_table.add_column("MRR_PT", justify="right")
    comp_table.add_column("MRR_AO", justify="right")
    comp_table.add_column("KW p", justify="right")
    comp_table.add_column("Spread", justify="right")

    for _, row in comparison_df.iterrows():
        comp_table.add_row(
            row["policy"],
            f"{row['avg_mrr']:.4f}",
            f"{row['avg_hit5']:.3f}",
            f"{row['mrr_br']:.4f}",
            f"{row['mrr_pt']:.4f}",
            f"{row['mrr_ao']:.4f}",
            f"{row['kw_p']:.4f}",
            f"{row['spread']:.4f}",
        )
    console.print(comp_table)

    # ---- Step 5: Analyze actions ----
    console.print("\n[bold green]Step 5: Analyzing DQN action distribution...[/bold green]")
    action_df = analyze_dqn_actions(qa_pairs, agent, lora_query_embs)

    # Print action stats
    console.print("\n[bold]Action distribution by country:[/bold]")
    for c in COUNTRIES:
        sub = action_df[action_df["country"] == c]
        counts = sub["action"].value_counts().sort_index()
        total = len(sub)
        parts = []
        for a in range(NUM_ACTIONS):
            count = counts.get(a, 0)
            pct = count / total * 100
            parts.append(f"A{a}={pct:.1f}%")
        console.print(f"  {c}: {' | '.join(parts)}")

    # ---- Step 6: Save results ----
    console.print("\n[bold green]Step 6: Saving results and generating plots...[/bold green]")

    # Save CSV
    csv_path = OUTPUT_DIR / "dqn_retriever_results.csv"
    comparison_df.to_csv(csv_path, index=False)
    console.print(f"  Results saved: {csv_path}")

    action_csv = OUTPUT_DIR / "dqn_action_analysis.csv"
    action_df.to_csv(action_csv, index=False)
    console.print(f"  Action analysis saved: {action_csv}")

    # Generate plots
    plot1 = plot_training_curve(history, OUTPUT_DIR)
    console.print(f"  Training curve: {plot1}")

    plot2 = plot_action_distribution(action_df, OUTPUT_DIR)
    console.print(f"  Action distribution: {plot2}")

    plot3 = plot_comparison(comparison_df, OUTPUT_DIR)
    console.print(f"  Comparison chart: {plot3}")

    # Copy to figures directory
    for plot_path in [plot1, plot2, plot3]:
        if plot_path and plot_path.exists():
            dest = FIGURES_DIR / plot_path.name
            shutil.copy2(plot_path, dest)
            console.print(f"  Copied to {dest}")

    # ---- Final summary ----
    console.print("\n[bold magenta]===== Final Summary =====[/bold magenta]")

    dqn_row = comparison_df[comparison_df["policy"] == "DQN Policy"].iloc[0]
    best_fixed_row = comparison_df[comparison_df["policy"] == "LoRA+CL a=0.9 (best fixed)"].iloc[0]
    oracle_row = comparison_df[comparison_df["policy"] == "Oracle (best per query)"].iloc[0]

    delta_mrr = dqn_row["avg_mrr"] - best_fixed_row["avg_mrr"]
    delta_pct = (delta_mrr / best_fixed_row["avg_mrr"] * 100) if best_fixed_row["avg_mrr"] > 0 else 0
    gap_to_oracle = oracle_row["avg_mrr"] - dqn_row["avg_mrr"]

    console.print(f"\n  DQN MRR:        {dqn_row['avg_mrr']:.4f}")
    console.print(f"  Best Fixed MRR: {best_fixed_row['avg_mrr']:.4f}")
    console.print(f"  Oracle MRR:     {oracle_row['avg_mrr']:.4f}")
    color = "green" if delta_mrr > 0 else "red"
    console.print(f"  [{color}]Delta vs Best Fixed: {delta_mrr:+.4f} ({delta_pct:+.2f}%)[/{color}]")
    console.print(f"  Gap to Oracle:  {gap_to_oracle:.4f}")

    console.print(f"\n  DQN Equity (KW p):        {dqn_row['kw_p']:.4f}")
    console.print(f"  Best Fixed Equity (KW p):  {best_fixed_row['kw_p']:.4f}")
    eq_color = "green" if dqn_row["kw_p"] > best_fixed_row["kw_p"] else "red"
    console.print(f"  [{eq_color}]Equity improvement: {dqn_row['kw_p'] - best_fixed_row['kw_p']:+.4f}[/{eq_color}]")

    console.print(f"\n[bold magenta]DQN Retriever Agent experiment complete![/bold magenta]\n")


if __name__ == "__main__":
    main()
