# utils.py
"""
Herramientas de análisis emocional + motor de propagación
---------------------------------------------------------

• EmotionAnalyzer  → vector de 10 dimensiones para un texto
• PropagationEngine
    · build(edges_df, states_df)
    · propagate(seed_user, message, max_steps)
      ↳ devuelve (vector_dict, log_serial)

El log_serial tiene esta forma:

  # Publicación inicial
  {
    "t": 0,
    "sender": null,
    "receiver": "user_2",
    "action": "publish",
    "vector_sent": [ ... ]                # ← vector del mensaje
  }

  # Interacciones posteriores
  {
    "t": 1,
    "sender": "user_2",
    "receiver": "user_53",
    "action": "modificar",
    "vector_sent": [ ... ],               # vector que recibió el usuario
    "sim_in": 0.66,
    "sim_out": 0.12,
    "state_in_before": [ ... ],
    "state_in_after":  [ ... ]
  }
"""

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

    # ---------- limpieza simple -------------------------------------
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

    # ---------- vector de 10 dimensiones ----------------------------
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


# ─────────────────────── MOTOR DE PROPAGACIÓN ──────────────────────
class PropagationEngine:
    def __init__(self) -> None:
        self.analyzer = EmotionAnalyzer()
        self.graph: nx.DiGraph | None = None

        self.state_in: Dict[str, np.ndarray] = {}
        self.state_out: Dict[str, np.ndarray] = {}
        self.alpha_u: Dict[str, float] = {}
        self.profile_u: Dict[str, str] = {}

    # ---------- Construcción de la red ------------------------------
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

    # ---------- Simulación ------------------------------------------
    def propagate(
        self, seed_user: str, message: str, max_steps: int = 4
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")

        vec_msg = self.analyzer.vector(message)
        vector_dict = {k: round(v, 3) for k, v in zip(EMOTION_COLS, vec_msg)}

        # agenda: (t, sender, receiver, vector_enviado)
        agenda = deque([(0, None, seed_user, vec_msg)])
        LOG: List[Dict[str, Any]] = []

        while agenda:
            t, sender, receiver, v = agenda.popleft()

            # ─── Publicación inicial ───────────────────────────────
            if sender is None:
                # NO se registra ninguna fila, solo se difunde a los seguidores
                for follower in self.graph.predecessors(receiver):
                    agenda.append((t, receiver, follower, v))
                continue  # ← salta al siguiente ciclo

            # ─── Resto de interacciones ────────────────────────────
            prev_in = self.state_in[receiver].copy()
            sim_in  = cosine_similarity([v], [prev_in])[0, 0]
            sim_out = cosine_similarity([v], [self.state_out[receiver]])[0, 0]
            action  = _decision(self.profile_u[receiver], sim_in, sim_out)

            alpha   = self.alpha_u[receiver]
            new_in  = _ema(prev_in, v, alpha)
            self.state_in[receiver] = new_in

            if action in {"reenviar", "modificar"} and t < max_steps:
                vec_to_send = v if action == "reenviar" else _ema(v, self.state_out[receiver], alpha)
                for follower in self.graph.predecessors(receiver):
                    agenda.append((t + 1, receiver, follower, vec_to_send))

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
                    "state_in_after":  np.round(new_in, 3).tolist(),
                }
            )           

        return vector_dict, LOG
