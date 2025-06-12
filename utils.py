from __future__ import annotations

import re
from collections import deque
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────── NLP y emociones ────────────────────────────
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from nrclex import NRCLex

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# ─────────────────────── ANALIZADOR EMOCIONAL ───────────────────────
class EmotionAnalyzer:
    def __init__(self) -> None:
        self.lem = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.labels = [
            "subjectivity",
            "polarity",
            "fear",
            "anger",
            "anticip",
            "trust",
            "surprise",
            "sadness",
            "disgust",
            "joy",
        ]

    def _clean(self, text: str) -> str:
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"RT[\s]+", "", text)
        text = re.sub(r"https?:\/\/\S+", "", text)
        text = re.sub(r":[ \s]+", "", text)
        text = re.sub(r"[\'\"]", "", text)
        text = re.sub(r"\.\.\.+", "", text)
        text = text.lower()

        tokens = nltk.word_tokenize(text)
        tokens = [
            self.lem.lemmatize(t)
            for t in tokens
            if t.isalpha() and t not in self.stop_words
        ]
        return " ".join(tokens)

    def vector(self, text: str) -> np.ndarray:
        clean = self._clean(text)
        blob = TextBlob(clean)
        subj, pol = blob.sentiment.subjectivity, blob.sentiment.polarity

        emo = NRCLex(clean).affect_frequencies
        extras = [
            emo.get("fear", 0.0),
            emo.get("anger", 0.0),
            emo.get("anticip", 0.0),
            emo.get("trust", 0.0),
            emo.get("surprise", 0.0),
            emo.get("sadness", 0.0),
            emo.get("disgust", 0.0),
            emo.get("joy", 0.0),
        ]
        return np.array([subj, pol] + extras, dtype=float)

    def as_dict(self, text: str) -> Dict[str, float]:
        return dict(zip(self.labels, self.vector(text).tolist()))

# ─────────────────────── AUXILIARES ────────────────────────────────
EMOTION_COLS: List[str] = [
    "subjectivity",
    "polarity",
    "fear",
    "anger",
    "anticip",
    "trust",
    "surprise",
    "sadness",
    "disgust",
    "joy",
]

def _col(prefix: str) -> List[str]:
    return [f"{prefix}_{c}" for c in EMOTION_COLS]

ALPHA_BY_PROFILE: Dict[str, float] = {
    "High-Credibility Informant": 0.30,
    "Emotionally-Driven Amplifier": 0.80,
    "Mobilisation-Oriented Catalyst": 0.70,
    "Emotionally Exposed Participant": 0.60,
}

def _decision(profile: str, sim_in: float, sim_out: float) -> str:
    if profile == "High-Credibility Informant":
        return (
            "reenviar"
            if (sim_in > 0.8 and sim_out > 0.7)
            else "modificar"
            if sim_in > 0.6
            else "ignorar"
        )
    if profile == "Emotionally-Driven Amplifier":
        return "reenviar" if sim_in > 0.4 else "modificar"
    if profile == "Mobilisation-Oriented Catalyst":
        return (
            "reenviar"
            if sim_in > 0.7
            else "modificar"
            if sim_in > 0.5
            else "ignorar"
        )
    if profile == "Emotionally Exposed Participant":
        return (
            "reenviar"
            if sim_in > 0.9
            else "modificar"
            if sim_in > 0.6
            else "ignorar"
        )
    raise ValueError(f"Perfil desconocido: {profile!r}")

def _ema(prev_vec: np.ndarray, new_vec: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new_vec + (1.0 - alpha) * prev_vec

# ─────────────────────── MOTOR DE PROPAGACIÓN ORIGINAL ─────────────
class PropagationEngine:
    def __init__(self) -> None:
        self.analyzer = EmotionAnalyzer()
        self.graph: nx.DiGraph | None = None
        self.state_in: Dict[str, np.ndarray] = {}
        self.state_out: Dict[str, np.ndarray] = {}
        self.alpha_u: Dict[str, float] = {}
        self.profile_u: Dict[str, str] = {}
        self.history: Dict[str, List[np.ndarray]] = {}  # Historial para state_in y state_out

    def build(
        self,
        edges_df: pd.DataFrame,
        states_df: pd.DataFrame,
        network_id: int | None = None,
    ) -> None:
        if network_id is not None and "network_id" in edges_df.columns:
            edges_df = edges_df.query("network_id == @network_id")

        self.graph = nx.from_pandas_edgelist(
            edges_df, source="source", target="target", create_using=nx.DiGraph
        )

        states_df = states_df.set_index("user_name")
        self.state_in.clear()
        self.state_out.clear()
        self.alpha_u.clear()
        self.profile_u.clear()
        self.history.clear()

        for user, row in states_df.iterrows():
            perfil = (
                "High-Credibility Informant"
                if row["cluster"] == 0
                else "Emotionally-Driven Amplifier"
                if row["cluster"] == 1
                else "Mobilisation-Oriented Catalyst"
                if row["cluster"] == 2
                else "Emotionally Exposed Participant"
            )
            self.state_in[user] = row[_col("in")].to_numpy(dtype=float)
            self.state_out[user] = row[_col("out")].to_numpy(dtype=float)
            self.alpha_u[user] = ALPHA_BY_PROFILE[perfil]
            self.profile_u[user] = perfil
            self.history[user] = [(self.state_in[user].copy(), self.state_out[user].copy())]

    def propagate(
        self, seed_user: str, message: str, max_steps: int = 4
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")

        vec_msg = self.analyzer.vector(message)
        vector_dict = {k: round(v, 3) for k, v in zip(EMOTION_COLS, vec_msg)}

        # Actualizar state_out del publicador inicial
        alpha = self.alpha_u[seed_user]
        prev_out = self.state_out[seed_user].copy()
        self.state_out[seed_user] = _ema(prev_out, vec_msg, alpha)
        self.history[seed_user].append((self.state_in[seed_user].copy(), self.state_out[seed_user].copy()))

        # agenda: (t, sender, receiver, vector_enviado)
        agenda = deque([(0, None, seed_user, vec_msg)])
        LOG: List[Dict[str, Any]] = []

        while agenda:
            t, sender, receiver, v = agenda.popleft()

            # ─── Publicación inicial ───────────────────────────────
            if sender is None:
                # Registrar publicación inicial en el log
                LOG.append(
                    {
                        "t": t,
                        "publisher": receiver,
                        "action": "publish",
                        "vector_sent": np.round(v, 3).tolist(),
                        "state_out_before": np.round(prev_out, 3).tolist(),
                        "state_out_after": np.round(self.state_out[receiver], 3).tolist(),
                    }
                )
                for follower in self.graph.predecessors(receiver):
                    agenda.append((t, receiver, follower, v))
                continue

            # ─── Resto de interacciones ────────────────────────────
            prev_in = self.state_in[receiver].copy()
            prev_out = self.state_out[receiver].copy()
            sim_in = cosine_similarity([v], [prev_in])[0, 0]
            sim_out = cosine_similarity([v], [self.state_out[receiver]])[0, 0]
            action = _decision(self.profile_u[receiver], sim_in, sim_out)

            alpha = self.alpha_u[receiver]
            new_in = _ema(prev_in, v, alpha)
            self.state_in[receiver] = new_in

            # Actualizar state_out si la acción es reenviar o modificar
            vec_to_send = v
            if action in {"reenviar", "modificar"}:
                vec_to_send = v if action == "reenviar" else _ema(v, prev_out, alpha)
                self.state_out[receiver] = _ema(prev_out, vec_to_send, alpha)

            # Actualizar historial
            self.history[receiver].append((self.state_in[receiver].copy(), self.state_out[receiver].copy()))

            # Registrar en el log
            LOG.append(
                {
                    "t": t,
                    "sender": sender,
                    "receiver": receiver,
                    "action": action,
                    "vector_sent": np.round(v, 3).tolist(),
                    "sim_in": round(sim_in, 3),
                    "sim_out": round(sim_out, 3),
                    "state_in_before": np.round(prev_in, 3).tolist(),
                    "state_in_after": np.round(new_in, 3).tolist(),
                    "state_out_before": np.round(prev_out, 3).tolist(),
                    "state_out_after": np.round(self.state_out[receiver], 3).tolist(),
                }
            )

            # Difundir a los seguidores
            if action in {"reenviar", "modificar"} and t < max_steps:
                for follower in self.graph.predecessors(receiver):
                    agenda.append((t + 1, receiver, follower, vec_to_send))

        return vector_dict, LOG

# ─────────────────────── MOTOR DE PROPAGACIÓN SIMPLE (RIP-DSN) ────
class SimplePropagationEngine:
    def __init__(self) -> None:
        self.graph: nx.DiGraph | None = None
        self.nodes: set = set()

    def build(
        self,
        links_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        network_id: int | None = None,
    ) -> None:
        if network_id is not None and "network_id" in links_df.columns:
            links_df = links_df.query("network_id == @network_id")
        if network_id is not None and "network_id" in nodes_df.columns:
            nodes_df = nodes_df.query("network_id == @network_id")

        self.graph = nx.from_pandas_edgelist(
            links_df, source="source", target="target", create_using=nx.DiGraph
        )
        self.nodes = set(nodes_df["node"].astype(str))

    def propagate(
        self, seed_user: str, message: str, max_steps: int = 4
    ) -> List[Dict[str, Any]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")
        if seed_user not in self.nodes:
            raise ValueError(f"Usuario inicial {seed_user} no encontrado en la red")

        # agenda: (t, sender, receiver)
        agenda = deque([(0, None, seed_user)])
        LOG: List[Dict[str, Any]] = []

        visited = set()

        while agenda:
            t, sender, receiver = agenda.popleft()

            # Evitar ciclos
            if receiver in visited:
                continue
            visited.add(receiver)

            # Registrar en el log
            LOG.append(
                {
                    "t": t,
                    "sender": sender,
                    "receiver": receiver,
                    "action": "publish" if sender is None else "forward",
                }
            )

            # Difundir a los seguidores
            if t < max_steps:
                for follower in self.graph.predecessors(receiver):
                    if follower in self.nodes:
                        agenda.append((t + 1, receiver, follower))

        return LOG