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
nltk.download("punkt_tab", quiet=True)

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
            if (sim_in > forward_threshold and sim_out > forward_threshold)
            else "modificar"
            if (sim_in > modify_threshold and sim_out > modify_threshold)
            else "ignorar"
        )
    if profile == "Emotionally-Driven Amplifier":
        return (
            "reenviar"
            if (sim_in > forward_threshold and sim_out > forward_threshold)
            else "modificar"
            if (sim_in > modify_threshold and sim_out > modify_threshold)
            else "ignorar"
        )
    if profile == "Mobilisation-Oriented Catalyst":
        return (
            "reenviar"
            if (sim_in > forward_threshold and sim_out > forward_threshold)
            else "modificar"
            if (sim_in > modify_threshold and sim_out > modify_threshold)
            else "ignorar"
        )
    if profile == "Emotionally Exposed Participant":
      return (
            "reenviar"
            if (sim_in > forward_threshold and sim_out > forward_threshold)
            else "modificar"
            if (sim_in > modify_threshold and sim_out > modify_threshold)
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
    # Usamos pandas para aprovechar ewm (EMA) y rolling (SMA)
    df = pd.DataFrame([prev_vec, new_vec], columns=EMOTION_COLS)

    if method == "ema":
        # alpha es obligatorio para EMA
        if alpha is None:
            raise ValueError("El método EMA requiere un valor de alpha")
        updated = df.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        return updated.to_numpy(dtype=float)
    elif method == "sma":
        # Media móvil simple 
        updated = df.rolling(window=2, min_periods=1).mean().iloc[-1]
        return updated.to_numpy(dtype=float)
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
        agenda = deque([(1, None, seed_user, vec_msg)])
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
        agenda = deque([(1, None, seed_user)])
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

# ─────────────────────── MOTORES DE PROPAGACIÓN SIR Y SIS ─────────────
class SIRPropagationEngine:
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
        self, seed_user: str, beta: float, gamma: float, max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")
        if seed_user not in self.nodes:
            raise ValueError(f"Usuario inicial {seed_user} no encontrado en la red")

        # Inicializar estados de nodos
        node_states = {node: 'susceptible' for node in self.nodes}
        node_states[seed_user] = 'infected'

        # Variables de simulación
        propagation_log = []
        current_infected = [seed_user]
        time_step = 1

        # Simulación de propagación SIR
        while current_infected and time_step < max_steps:
            new_infected = []

            # Fase 1: Propagación de infección a vecinos susceptibles
            for infected_id in current_infected:
                # Obtener vecinos susceptibles (predecesores en el grafo dirigido)
                susceptible_neighbors = [
                    neighbor for neighbor in self.graph.predecessors(infected_id)
                    if neighbor in self.nodes and node_states[neighbor] == 'susceptible'
                ]

                for neighbor in susceptible_neighbors:
                    if np.random.random() < beta:
                        # Infectar nodo susceptible
                        node_states[neighbor] = 'infected'
                        new_infected.append(neighbor)
                        
                        # Registrar evento de infección
                        propagation_log.append({
                            "t": time_step,
                            "sender": infected_id,
                            "receiver": neighbor,
                            "action": "infect",
                            "state": "infected"
                        })

            # Fase 2: Verificar recuperación de infectados
            recovered_this_step = []
            for infected_id in current_infected:
                if np.random.random() < gamma:
                    node_states[infected_id] = 'recovered'
                    recovered_this_step.append(infected_id)
                    
                    # Registrar evento de recuperación
                    propagation_log.append({
                        "t": time_step,
                        "sender": infected_id,
                        "receiver": infected_id,
                        "action": "recover",
                        "state": "recovered"
                    })

            # Actualizar para el siguiente paso de tiempo
            current_infected = [node for node in current_infected + new_infected 
                              if node not in recovered_this_step]
            time_step += 1

        return propagation_log

class SISPropagationEngine:
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
        self, seed_user: str, beta: float, gamma: float, max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")
        if seed_user not in self.nodes:
            raise ValueError(f"Usuario inicial {seed_user} no encontrado en la red")

        # Inicializar estados de nodos
        node_states = {node: 'susceptible' for node in self.nodes}
        node_states[seed_user] = 'infected'

        # Variables de simulación
        propagation_log = []
        current_infected = [seed_user]
        time_step = 1

        # Simulación de propagación SIS
        while current_infected and time_step < max_steps:
            new_infected = []

            # Fase 1: Propagación de infección a vecinos susceptibles
            for infected_id in current_infected:
                # Obtener vecinos susceptibles (predecesores en el grafo dirigido)
                susceptible_neighbors = [
                    neighbor for neighbor in self.graph.predecessors(infected_id)
                    if neighbor in self.nodes and node_states[neighbor] == 'susceptible'
                ]

                for neighbor in susceptible_neighbors:
                    if np.random.random() < beta:
                        # Infectar nodo susceptible
                        node_states[neighbor] = 'infected'
                        new_infected.append(neighbor)
                        
                        # Registrar evento de infección
                        propagation_log.append({
                            "t": time_step,
                            "sender": infected_id,
                            "receiver": neighbor,
                            "action": "infect",
                            "state": "infected"
                        })

            # Fase 2: Verificar recuperación de infectados (vuelven a susceptibles)
            recovered_this_step = []
            for infected_id in current_infected:
                if np.random.random() < gamma:
                    node_states[infected_id] = 'susceptible'
                    recovered_this_step.append(infected_id)
                    
                    # Registrar evento de recuperación
                    propagation_log.append({
                        "t": time_step,
                        "sender": infected_id,
                        "receiver": infected_id,
                        "action": "recover",
                        "state": "susceptible"
                    })

            # Actualizar para el siguiente paso de tiempo
            current_infected = [node for node in current_infected + new_infected 
                              if node not in recovered_this_step]
            time_step += 1

        return propagation_log

class RWSIRPropagationEngine:
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
        self, seed_user: str, beta: float, gamma: float, max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")
        if seed_user not in self.nodes:
            raise ValueError(f"Usuario inicial {seed_user} no encontrado en la red")

        # Inicializar estados de nodos
        node_states = {node: 'susceptible' for node in self.nodes}
        node_states[seed_user] = 'infected'

        # Variables de simulación
        propagation_log = []
        current_infected = [seed_user]
        time_step = 1

        # Simulación de propagación SIR en red del mundo real
        while current_infected and time_step < max_steps:
            new_infected = []

            # Fase 1: Propagación de infección a vecinos susceptibles
            for infected_id in current_infected:
                # Obtener vecinos susceptibles (predecesores en el grafo dirigido)
                susceptible_neighbors = [
                    neighbor for neighbor in self.graph.predecessors(infected_id)
                    if neighbor in self.nodes and node_states[neighbor] == 'susceptible'
                ]

                for neighbor in susceptible_neighbors:
                    if np.random.random() < beta:
                        # Infectar nodo susceptible
                        node_states[neighbor] = 'infected'
                        new_infected.append(neighbor)
                        
                        # Registrar evento de infección
                        propagation_log.append({
                            "t": time_step,
                            "sender": infected_id,
                            "receiver": neighbor,
                            "action": "infect",
                            "state": "infected"
                        })

            # Fase 2: Verificar recuperación de infectados
            recovered_this_step = []
            for infected_id in current_infected:
                if np.random.random() < gamma:
                    node_states[infected_id] = 'recovered'
                    recovered_this_step.append(infected_id)
                    
                    # Registrar evento de recuperación
                    propagation_log.append({
                        "t": time_step,
                        "sender": infected_id,
                        "receiver": infected_id,
                        "action": "recover",
                        "state": "recovered"
                    })

            # Actualizar para el siguiente paso de tiempo
            current_infected = [node for node in current_infected + new_infected 
                              if node not in recovered_this_step]
            time_step += 1

        return propagation_log

class RWSISPropagationEngine:
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
        self, seed_user: str, beta: float, gamma: float, max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        if self.graph is None:
            raise RuntimeError("Primero llama a build()")
        if seed_user not in self.nodes:
            raise ValueError(f"Usuario inicial {seed_user} no encontrado en la red")

        # Inicializar estados de nodos
        node_states = {node: 'susceptible' for node in self.nodes}
        node_states[seed_user] = 'infected'

        # Variables de simulación
        propagation_log = []
        current_infected = [seed_user]
        time_step = 1

        # Simulación de propagación SIS en red del mundo real
        while current_infected and time_step < max_steps:
            new_infected = []

            # Fase 1: Propagación de infección a vecinos susceptibles
            for infected_id in current_infected:
                # Obtener vecinos susceptibles (predecesores en el grafo dirigido)
                susceptible_neighbors = [
                    neighbor for neighbor in self.graph.predecessors(infected_id)
                    if neighbor in self.nodes and node_states[neighbor] == 'susceptible'
                ]

                for neighbor in susceptible_neighbors:
                    if np.random.random() < beta:
                        # Infectar nodo susceptible
                        node_states[neighbor] = 'infected'
                        new_infected.append(neighbor)
                        
                        # Registrar evento de infección
                        propagation_log.append({
                            "t": time_step,
                            "sender": infected_id,
                            "receiver": neighbor,
                            "action": "infect",
                            "state": "infected"
                        })

            # Fase 2: Verificar recuperación de infectados (vuelven a susceptibles)
            recovered_this_step = []
            for infected_id in current_infected:
                if np.random.random() < gamma:
                    node_states[infected_id] = 'susceptible'
                    recovered_this_step.append(infected_id)
                    
                    # Registrar evento de recuperación
                    propagation_log.append({
                        "t": time_step,
                        "sender": infected_id,
                        "receiver": infected_id,
                        "action": "recover",
                        "state": "susceptible"
                    })

            # Actualizar para el siguiente paso de tiempo
            current_infected = [node for node in current_infected + new_infected 
                              if node not in recovered_this_step]
            time_step += 1

        return propagation_log

def calculate_alcance_final(propagation_log: List[Dict[str, Any]]) -> int:
    """
    Calcula el alcance final de una propagación contando el número de nodos únicos
    que participaron en la propagación (infectados, recuperados, o que recibieron mensajes).
    
    Args:
        propagation_log: Lista de eventos de propagación
        
    Returns:
        Número de nodos únicos que participaron en la propagación
    """
    unique_nodes = set()
    
    for event in propagation_log:
        # Agregar nodos que enviaron mensajes
        if 'sender' in event and event['sender'] is not None:
            unique_nodes.add(event['sender'])
        
        # Agregar nodos que recibieron mensajes
        if 'receiver' in event:
            unique_nodes.add(event['receiver'])
        
        # Agregar nodos que publicaron mensajes
        if 'publisher' in event:
            unique_nodes.add(event['publisher'])
    
    return len(unique_nodes)

def calculate_t_pico(propagation_log: List[Dict[str, Any]], method: str = "sir") -> Dict[int, int]:
    """
    Calcula t_pico: el número de nodos infectados/activos en cada paso de tiempo.
    
    Para modelos SIR/SIS: cuenta nodos infectados en cada paso t
    Para RIP-DSN: cuenta nodos que reenvían o modifican en cada paso t
    
    Args:
        propagation_log: Lista de eventos de propagación
        method: Tipo de propagación ("sir", "sis", "rip-dsn", "emotion")
        
    Returns:
        Diccionario con {paso_tiempo: numero_nodos_activos}
    """
    print(f"\n=== CALCULANDO T_PICO PARA MÉTODO: {method} ===")
    print(f"Total de eventos en el log: {len(propagation_log)}")
    
    t_pico = {}
    
    if method in ["sir", "sis"]:
        print(f"\n--- Procesando modelo {method.upper()} ---")
        # Para SIR/SIS: contar nodos infectados en cada paso de tiempo
        infected_by_time = {}
        
        for i, event in enumerate(propagation_log):
            t = event.get('t', 0)
            action = event.get('action', '')
            receiver = event.get('receiver', '')
            sender = event.get('sender', '')
            
            print(f"Evento {i+1}: t={t}, action='{action}', sender='{sender}', receiver='{receiver}'")
            
            if action == 'infect':
                # Nuevo nodo infectado
                if t not in infected_by_time:
                    infected_by_time[t] = set()
                    print(f"  → Creando conjunto para tiempo t={t}")
                infected_by_time[t].add(receiver)
                print(f"  → Agregando {receiver} a infectados en t={t}")
                print(f"  → Infectados en t={t}: {list(infected_by_time[t])}")
            elif action == 'recover':
                # Nodo recuperado (solo para SIR, en SIS vuelve a susceptible)
                print(f"  → {receiver} se recupera en t={t} (no se cuenta como infectado)")
                if method == "sir":
                    # En SIR, los recuperados no se cuentan como infectados
                    pass
                else:  # SIS
                    # En SIS, los recuperados vuelven a susceptibles, no se cuentan
                    pass
            else:
                print(f"  → Acción '{action}' ignorada para conteo de infectados")
        
        print(f"\n--- Resumen de infectados por tiempo ---")
        # Calcular el número total de infectados en cada paso de tiempo
        for t in sorted(infected_by_time.keys()):
            count = len(infected_by_time[t])
            t_pico[str(t)] = count
            print(f"Tiempo t={t}: {count} nodos infectados {list(infected_by_time[t])}")
            
    elif method in ["rip-dsn", "emotion"]:
        print(f"\n--- Procesando modelo {method.upper()} ---")
        # Para RIP-DSN y propagación emocional: contar nodos que reenvían o modifican
        active_by_time = {}
        
        for i, event in enumerate(propagation_log):
            t = event.get('t', 0)
            action = event.get('action', '')
            receiver = event.get('receiver', '')
            sender = event.get('sender', '')
            publisher = event.get('publisher', '')
            
            print(f"Evento {i+1}: t={t}, action='{action}', sender='{sender}', receiver='{receiver}', publisher='{publisher}'")
            
            if action in ['reenviar', 'modificar', 'forward', 'ignorar']:
                # Nodo que reenvía, modifica o ignora
                if t not in active_by_time:
                    active_by_time[t] = set()
                    print(f"  → Creando conjunto para tiempo t={t}")
                active_by_time[t].add(receiver)
                print(f"  → Agregando {receiver} a activos en t={t} (acción: {action})")
                print(f"  → Activos en t={t}: {list(active_by_time[t])}")
            elif action == 'publish':
                # Para t_pico, no contamos la publicación inicial, solo las acciones de respuesta
                print(f"  → Acción 'publish' - no se cuenta para t_pico (solo acciones de respuesta)")
            else:
                print(f"  → Acción '{action}' ignorada para conteo de activos")
        
        print(f"\n--- Resumen de activos por tiempo ---")
        # Calcular el número total de nodos activos en cada paso de tiempo
        for t in sorted(active_by_time.keys()):
            count = len(active_by_time[t])
            t_pico[str(t)] = count
            print(f"Tiempo t={t}: {count} nodos activos {list(active_by_time[t])}")
    
    print(f"\n=== RESULTADO FINAL T_PICO ===")
    for t_str in sorted(t_pico.keys(), key=int):
        print(f"t_pico['{t_str}'] = {t_pico[t_str]}")
    print("=" * 50)
    
    return t_pico

def calculate_pct_modificar(propagation_log: List[Dict[str, Any]], total_nodes: int) -> float:
    """
    Calcula la proporción de nodos que modificaron al menos un mensaje durante la propagación.
    
    Args:
        propagation_log: Lista de eventos de propagación
        total_nodes: Número total de nodos en la red
        
    Returns:
        Proporción de nodos que modificaron mensajes (0.0 a 1.0)
    """
    print(f"\n=== CALCULANDO PCT_MODIFICAR ===")
    print(f"Total de nodos en la red: {total_nodes}")
    print(f"Total de eventos en el log: {len(propagation_log)}")
    print()
    
    nodes_that_modified = set()
    
    print("--- ANÁLISIS DE EVENTOS ---")
    for i, event in enumerate(propagation_log):
        action = event.get('action', '')
        receiver = event.get('receiver', '')
        sender = event.get('sender', '')
        publisher = event.get('publisher', '')
        t = event.get('t', 0)
        
        print(f"Evento {i+1}: t={t}, sender='{sender}', receiver='{receiver}', publisher='{publisher}', action='{action}'")
        
        if action == 'modificar':
            nodes_that_modified.add(receiver)
            print(f"  → NODO {receiver} MODIFICÓ mensaje")
        else:
            print(f"  → Acción '{action}' - no es modificar")
    
    print(f"\n--- RESULTADO PCT_MODIFICAR ---")
    print(f"Nodos que modificaron: {sorted(list(nodes_that_modified))}")
    print(f"Cantidad de nodos que modificaron: {len(nodes_that_modified)}")
    
    if total_nodes == 0:
        print("Total de nodos es 0, retornando 0.0")
        return 0.0
    
    percentage = len(nodes_that_modified) / total_nodes
    result = round(percentage, 4)
    print(f"Proporción: ({len(nodes_that_modified)} / {total_nodes}) = {result}")
    print("=" * 50)
    
    return result

def calculate_pct_reenviar(propagation_log: List[Dict[str, Any]], total_nodes: int) -> float:
    """
    Calcula la proporción de nodos que reenviaron al menos un mensaje durante la propagación.
    
    Args:
        propagation_log: Lista de eventos de propagación
        total_nodes: Número total de nodos en la red
        
    Returns:
        Proporción de nodos que reenviaron mensajes (0.0 a 1.0)
    """
    print(f"\n=== CALCULANDO PCT_REENVIAR ===")
    print(f"Total de nodos en la red: {total_nodes}")
    print(f"Total de eventos en el log: {len(propagation_log)}")
    print()
    
    nodes_that_forwarded = set()
    
    print("--- ANÁLISIS DE EVENTOS ---")
    for i, event in enumerate(propagation_log):
        action = event.get('action', '')
        receiver = event.get('receiver', '')
        sender = event.get('sender', '')
        publisher = event.get('publisher', '')
        t = event.get('t', 0)
        
        print(f"Evento {i+1}: t={t}, sender='{sender}', receiver='{receiver}', publisher='{publisher}', action='{action}'")
        
        if action in ['reenviar', 'forward']:
            nodes_that_forwarded.add(receiver)
            print(f"  → NODO {receiver} REENVIÓ mensaje (acción: {action})")
        else:
            print(f"  → Acción '{action}' - no es reenvío")
    
    print(f"\n--- RESULTADO PCT_REENVIAR ---")
    print(f"Nodos que reenviaron: {sorted(list(nodes_that_forwarded))}")
    print(f"Cantidad de nodos que reenviaron: {len(nodes_that_forwarded)}")
    
    if total_nodes == 0:
        print("Total de nodos es 0, retornando 0.0")
        return 0.0
    
    percentage = len(nodes_that_forwarded) / total_nodes
    result = round(percentage, 4)
    print(f"Proporción: ({len(nodes_that_forwarded)} / {total_nodes}) = {result}")
    print("=" * 50)
    
    return result

def calculate_pct_ignorar(propagation_log: List[Dict[str, Any]], total_nodes: int, alcance_final: int) -> float:
    """
    Calcula el porcentaje de nodos que ignoraron mensajes (no participaron en la propagación).
    
    La fórmula correcta es: (total_nodes - alcance_final) / total_nodes
    
    Args:
        propagation_log: Lista de eventos de propagación (no se usa, solo para compatibilidad)
        total_nodes: Número total de nodos en la red
        alcance_final: Número de nodos que participaron en la propagación (recibieron el mensaje)
        
    Returns:
        Proporción de nodos que ignoraron mensajes (0.0 a 1.0)
    """
    print(f"\n=== CALCULANDO PCT_IGNORAR (FÓRMULA CORREGIDA) ===")
    print(f"Total de nodos en la red: {total_nodes}")
    print(f"Alcance final (nodos que participaron): {alcance_final}")
    print(f"Nodos que NO participaron (ignoraron): {total_nodes - alcance_final}")
    
    if total_nodes == 0:
        print("Total de nodos es 0, retornando 0.0")
        print("=" * 50)
        return 0.0
    
    # Fórmula corregida: nodos que no participaron / total de nodos
    pct_ignorar = (total_nodes - alcance_final) / total_nodes
    
    # Asegurar que el resultado esté en el rango [0, 1]
    pct_ignorar = max(0.0, min(1.0, pct_ignorar))
    
    result = round(pct_ignorar, 4)
    print(f"Fórmula: ({total_nodes} - {alcance_final}) / {total_nodes} = {result}")
    print(f"Porcentaje: {result * 100:.2f}%")
    print("=" * 50)
    
    return result

def calculate_new_t(propagation_log: List[Dict[str, Any]], method: str = "rip-dsn") -> Dict[int, int]:
    """
    Calcula new_t: el número de nodos que participan por primera vez en cada paso de tiempo.
    
    A diferencia de t_pico, new_t no cuenta nodos que ya participaron en pasos anteriores.
    
    Args:
        propagation_log: Lista de eventos de propagación
        method: Tipo de propagación ("sir", "sis", "rip-dsn", "emotion")
        
    Returns:
        Diccionario con {paso_tiempo: numero_nodos_nuevos}
    """
    print(f"\n=== CALCULANDO NEW_T PARA MÉTODO: {method} ===")
    print(f"Total de eventos en el log: {len(propagation_log)}")
    
    new_t = {}
    nodes_that_participated = set()  # Nodos que ya participaron en pasos anteriores
    
    if method in ["sir", "sis"]:
        print(f"\n--- Procesando modelo {method.upper()} para new_t ---")
        # Para SIR/SIS: contar nodos infectados por primera vez en cada paso de tiempo
        new_infected_by_time = {}
        
        for i, event in enumerate(propagation_log):
            t = event.get('t', 0)
            action = event.get('action', '')
            receiver = event.get('receiver', '')
            sender = event.get('sender', '')
            
            print(f"Evento {i+1}: t={t}, action='{action}', sender='{sender}', receiver='{receiver}'")
            
            if action == 'infect':
                # Solo contar si el nodo no había participado antes
                if receiver not in nodes_that_participated:
                    if t not in new_infected_by_time:
                        new_infected_by_time[t] = set()
                        print(f"  → Creando conjunto para tiempo t={t}")
                    new_infected_by_time[t].add(receiver)
                    nodes_that_participated.add(receiver)
                    print(f"  → Agregando {receiver} a nuevos infectados en t={t} (primera participación)")
                    print(f"  → Nuevos infectados en t={t}: {list(new_infected_by_time[t])}")
                else:
                    print(f"  → {receiver} ya participó antes, no se cuenta en new_t")
            else:
                print(f"  → Acción '{action}' ignorada para conteo de nuevos infectados")
        
        print(f"\n--- Resumen de nuevos infectados por tiempo ---")
        # Calcular el número total de nuevos infectados en cada paso de tiempo
        for t in sorted(new_infected_by_time.keys()):
            count = len(new_infected_by_time[t])
            new_t[str(t)] = count
            print(f"Tiempo t={t}: {count} nuevos infectados {list(new_infected_by_time[t])}")
            
    elif method in ["rip-dsn", "emotion"]:
        print(f"\n--- Procesando modelo {method.upper()} para new_t ---")
        # Para RIP-DSN y propagación emocional: contar nodos que participan por primera vez
        new_active_by_time = {}
        
        for i, event in enumerate(propagation_log):
            t = event.get('t', 0)
            action = event.get('action', '')
            receiver = event.get('receiver', '')
            sender = event.get('sender', '')
            publisher = event.get('publisher', '')
            
            print(f"Evento {i+1}: t={t}, action='{action}', sender='{sender}', receiver='{receiver}', publisher='{publisher}'")
            
            if action in ['reenviar', 'modificar', 'forward', 'ignorar']:
                # Solo contar si el nodo no había participado antes
                if receiver not in nodes_that_participated:
                    if t not in new_active_by_time:
                        new_active_by_time[t] = set()
                        print(f"  → Creando conjunto para tiempo t={t}")
                    new_active_by_time[t].add(receiver)
                    nodes_that_participated.add(receiver)
                    print(f"  → Agregando {receiver} a nuevos activos en t={t} (primera participación, acción: {action})")
                    print(f"  → Nuevos activos en t={t}: {list(new_active_by_time[t])}")
                else:
                    print(f"  → {receiver} ya participó antes, no se cuenta en new_t")
            elif action == 'publish':
                # Para new_t, no contamos la publicación inicial
                print(f"  → Acción 'publish' - no se cuenta para new_t (solo acciones de respuesta)")
            else:
                print(f"  → Acción '{action}' ignorada para conteo de nuevos activos")
        
        print(f"\n--- Resumen de nuevos activos por tiempo ---")
        # Calcular el número total de nodos nuevos activos en cada paso de tiempo
        for t in sorted(new_active_by_time.keys()):
            count = len(new_active_by_time[t])
            new_t[str(t)] = count
            print(f"Tiempo t={t}: {count} nuevos activos {list(new_active_by_time[t])}")
    
    print(f"\n=== RESULTADO FINAL NEW_T ===")
    for t_str in sorted(new_t.keys(), key=int):
        print(f"new_t['{t_str}'] = {new_t[t_str]}")
    print("=" * 50)
    
    return new_t

def calculate_t_max(t_pico: Dict[str, int]) -> int:
    """
    Calcula t_max: el paso de tiempo donde ocurre el valor máximo de t_pico.
    
    Args:
        t_pico: Diccionario con {paso_tiempo: numero_nodos_activos}
        
    Returns:
        Paso de tiempo donde ocurre el valor máximo
    """
    print(f"\n=== CALCULANDO T_MAX ===")
    print(f"t_pico recibido: {t_pico}")
    
    if not t_pico:
        print("t_pico está vacío, retornando 0")
        return 0
    
    # Encontrar el paso de tiempo con el valor máximo
    max_time = max(t_pico.keys(), key=lambda t: t_pico[t])
    max_value = t_pico[max_time]
    
    print(f"Valores por tiempo: {[(t, v) for t, v in sorted(t_pico.items(), key=lambda x: int(x[0]))]}")
    print(f"Valor máximo: {max_value} en tiempo t={max_time}")
    print(f"t_max = {max_time} (paso de tiempo donde ocurre el máximo)")
    print("=" * 50)
    
    return int(max_time)