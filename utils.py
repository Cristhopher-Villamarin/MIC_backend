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

DEFAULT_ALPHA_BY_PROFILE: Dict[str, float] = {
    "High-Credibility Informant": 0.3,
    "Emotionally-Driven Amplifier": 0.8,
    "Mobilisation-Oriented Catalyst": 0.7,
    "Emotionally Exposed Participant": 0.6,
}

DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "High-Credibility Informant": {"forward": 0.8, "modify": 0.2, "ignore": 0.05},
    "Emotionally-Driven Amplifier": {"forward": 0.95, "modify": 0.6, "ignore": 0.1},
    "Mobilisation-Oriented Catalyst": {"forward": 0.6, "modify": 0.7, "ignore": 0.3},
    "Emotionally Exposed Participant": {"forward": 0.3, "modify": 0.4, "ignore": 0.7},
}

def _decision(profile: str, sim_in: float, sim_out: float, thresholds: Dict[str, float]) -> str:
    forward_threshold = thresholds.get("forward", DEFAULT_THRESHOLDS[profile]["forward"])
    modify_threshold = thresholds.get("modify", DEFAULT_THRESHOLDS[profile]["modify"])
    
    if profile == "High-Credibility Informant":
        return (
            "reenviar"
            if (sim_in > forward_threshold and sim_out > 0.7)
            else "modificar"
            if sim_in > modify_threshold
            else "ignorar"
        )
    if profile == "Emotionally-Driven Amplifier":
        return "reenviar" if sim_in > forward_threshold else "modificar" if sim_in > modify_threshold else "ignorar"
    if profile == "Mobilisation-Oriented Catalyst":
        return (
            "reenviar"
            if sim_in > forward_threshold
            else "modificar"
            if sim_in > modify_threshold
            else "ignorar"
        )
    if profile == "Emotionally Exposed Participant":
        return (
            "reenviar"
            if sim_in > forward_threshold
            else "modificar"
            if sim_in > modify_threshold
            else "ignorar"
        )
    raise ValueError(f"Perfil desconocido: {profile!r}")

def _update_vector(prev_vec: np.ndarray, new_vec: np.ndarray, alpha: float, method: str) -> np.ndarray:
    """
    Actualiza un vector usando media móvil exponencial (EMA) o simple (SMA).
    
    Args:
        prev_vec: Vector previo (estado anterior).
        new_vec: Vector nuevo (estado actual).
        alpha: Factor de suavizado para EMA.
        method: Método de actualización ('ema' o 'sma').
    
    Returns:
        Vector actualizado.
    """
    if method == "ema":
        return alpha * new_vec + (1.0 - alpha) * prev_vec
    elif method == "sma":
        return (prev_vec + new_vec) / 2.0
    else:
        raise ValueError(f"Método no reconocido: {method}. Use 'ema' o 'sma'.")

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
        self.thresholds_u: Dict[str, Dict[str, float]] = {}

    def build(
        self,
        edges_df: pd.DataFrame,
        states_df: pd.DataFrame,
        network_id: int | None = None,
        thresholds: Dict[str, Dict[str, float]] = {}
    ) -> None:
        if network_id is not None and "network_id" in edges_df.columns:
            edges_df = edges_df.query("network_id == @network_id")

        # Filtrar aristas donde source == target
        edges_df = edges_df[edges_df['source'] != edges_df['target']]

        print("edges_df:", edges_df.head().to_dict())
        print("states_df:", states_df.head().to_dict())

        self.graph = nx.from_pandas_edgelist(
            edges_df, source="source", target="target", create_using=nx.DiGraph
        )

        states_df = states_df.set_index("user_name")
        self.state_in.clear()
        self.state_out.clear()
        self.alpha_u.clear()
        self.profile_u.clear()
        self.history.clear()
        self.thresholds_u.clear()

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
            self.alpha_u[user] = thresholds.get(perfil, {}).get("alpha", DEFAULT_ALPHA_BY_PROFILE[perfil])
            self.profile_u[user] = perfil
            self.thresholds_u[user] = thresholds.get(perfil, DEFAULT_THRESHOLDS[perfil])
            self.history[user] = [(self.state_in[user].copy(), self.state_out[user].copy())]

    def propagate(
        self, seed_user: str, message: str, max_steps: int = 4, method: str = "ema", custom_vector: np.ndarray | None = None
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")

        # Use custom_vector if provided, otherwise analyze the message
        vec_msg = custom_vector if custom_vector is not None else self.analyzer.vector(message)
        vector_dict = {k: round(v, 3) for k, v in zip(EMOTION_COLS, vec_msg)}

        # Actualizar state_out del publicador inicial
        alpha = self.alpha_u[seed_user]
        prev_out = self.state_out[seed_user].copy()
        self.state_out[seed_user] = _update_vector(prev_out, vec_msg, alpha, method)
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
            action = _decision(self.profile_u[receiver], sim_in, sim_out, self.thresholds_u[receiver])

            alpha = self.alpha_u[receiver]
            new_in = _update_vector(prev_in, v, alpha, method)
            self.state_in[receiver] = new_in

            # Actualizar state_out si la acción es reenviar o modificar
            vec_to_send = v
            if action in {"reenviar", "modificar"}:
                vec_to_send = v if action == "reenviar" else _update_vector(v, prev_out, alpha, method)
                self.state_out[receiver] = _update_vector(prev_out, vec_to_send, alpha, method)

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

        # Usar un diccionario para rastrear el número de veces que un nodo recibe el mensaje
        received_count = {node: 0 for node in self.nodes}

        while agenda:
            t, sender, receiver = agenda.popleft()

            # Incrementar el conteo de recepción
            received_count[receiver] += 1
            if received_count[receiver] > 1:
                LOG.append(
                    {
                        "t": t,
                        "sender": sender,
                        "receiver": receiver,
                        "action": "forward (repeated)",
                        "note": f"Received {received_count[receiver]} times",
                    }
                )
                continue  # Evitar propagación repetida

            # Registrar en el log
            LOG.append(
                {
                    "t": t,
                    "sender": sender,
                    "receiver": receiver,
                    "action": "publish" if sender is None else "forward",
                }
            )

            # Difundir solo a los predecesores (seguidores)
            if t < max_steps:
                for follower in self.graph.predecessors(receiver):
                    if follower in self.nodes and not any(
                        l["sender"] == receiver and l["receiver"] == follower and l["action"] == "forward"
                        for l in LOG
                    ):
                        agenda.append((t + 1, receiver, follower))

        return LOG