import nltk
nltk.download("punkt", quiet=True)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from generate_vectors import generar_datos_sinteticos_cargado, cargar_modelo_y_escalador
from utils import EmotionAnalyzer, PropagationEngine, SimplePropagationEngine, SIRPropagationEngine, SISPropagationEngine, RWSIRPropagationEngine, RWSISPropagationEngine, calculate_alcance_final, calculate_t_pico, calculate_t_max, calculate_pct_modificar, calculate_pct_reenviar, calculate_pct_ignorar
from pymongo import MongoClient
from datetime import datetime
import uuid

app = FastAPI(
    title="Backend · Propagación Emocional",
    description="Endpoints de prueba para propagar mensajes en una red y generar vectores sintéticos",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = EmotionAnalyzer()               # ⇠ /analyze
engine = PropagationEngine()              # ⇠ /propagate (original)
simple_engine = SimplePropagationEngine()  # ⇠ /propagate (RIP-DSN)
sir_engine = SIRPropagationEngine()       # ⇠ /propagate-sir
sis_engine = SISPropagationEngine()       # ⇠ /propagate-sis
rw_sir_engine = RWSIRPropagationEngine()  # ⇠ /propagate-rw-sir
rw_sis_engine = RWSISPropagationEngine()  # ⇠ /propagate-rw-sis

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"  # Replace with your MongoDB URI
DB_NAME = "emotional_propagation"
COLLECTION_NAME = "propagation_logs"

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# Cargar modelo VAE, escalador y metadatos al iniciar el servidor
try:
    vae_model, scaler, feature_columns, cluster_column = cargar_modelo_y_escalador(
        model_path='vae_model.keras',
        scaler_path='scaler.pkl',
        metadata_path='model_metadata.json'
    )
except Exception as e:
    print(f"Error al cargar el modelo, escalador o metadatos: {e}")
    raise Exception("No se pudo inicializar el modelo VAE, escalador o metadatos")

# ───────────────────────── ENDPOINTS ───────────────────────────────────
@app.post("/analyze")
async def analyze(text: str = Form(...)):
    """
    Devuelve el vector emocional de un texto (10 dimensiones).
    """
    try:
        return {
            "vector": analyzer.as_dict(text),
            "message": "Texto analizado correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al analizar el texto: {str(e)}")

@app.post("/analyze-message")
async def analyze_message(
    message: str = Form(...),
    custom_vector: str = Form(None, description="JSON con vector emocional personalizado")
):
    """
    Analiza el mensaje y devuelve su vector emocional. Si se proporciona un custom_vector, lo usa.
    """
    try:
        if custom_vector:
            try:
                vector = json.loads(custom_vector)
                if not isinstance(vector, dict):
                    raise ValueError("El vector personalizado debe ser un diccionario")
                complete_vector = {key: vector.get(key, 0.0) for key in analyzer.labels}
                return {
                    "vector": complete_vector,
                    "message": "Vector personalizado recibido correctamente",
                }
            except json.JSONDecodeError:
                raise HTTPException(400, detail="El custom_vector debe ser un JSON válido")
            except ValueError as ve:
                raise HTTPException(400, detail=str(ve))
        return {
            "vector": analyzer.as_dict(message),
            "message": "Mensaje analizado correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al analizar el mensaje: {str(e)}")

@app.post("/propagate")
async def propagate(
    seed_user: str = Form(..., description="Usuario origen"),
    message: str = Form(..., description="Mensaje a propagar"),
    csv_file: UploadFile = File(None, description="CSV con aristas"),
    xlsx_file: UploadFile = File(None, description="Excel con estados"),
    nodes_csv_file: UploadFile = File(None, description="CSV con nodos"),
    links_csv_file: UploadFile = File(None, description="CSV con relaciones"),
    max_steps: int = Form(4, ge=1, le=10),
    method: str = Form("ema", description="Método de actualización: 'ema' o 'sma'"),
    thresholds: str = Form("{}", description="JSON con umbrales y alphas por perfil"),
    custom_vector: str = Form(None, description="JSON con vector emocional personalizado"),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    cluster_filtering: str = Form(..., description="Filtrado de clúster"),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("barabasi-albert", description="Tipo de red"),
    metodo: str = Form("RIP-DSN", description="Método de propagación")
):
    try:
        thresholds_dict = json.loads(thresholds) if thresholds else {}
        if csv_file and xlsx_file and not (nodes_csv_file or links_csv_file):
            if method not in ["ema", "sma"]:
                raise HTTPException(400, detail="El método debe ser 'ema' o 'sma'")
            edges_df = pd.read_csv(csv_file.file)
            states_df = pd.read_excel(xlsx_file.file)
            engine.build(edges_df, states_df, thresholds=thresholds_dict)
            # Verificar si seed_user está en el grafo
            if seed_user not in engine.graph.nodes:
                raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
            if custom_vector:
                try:
                    vector_dict = json.loads(custom_vector)
                    if not isinstance(vector_dict, dict):
                        raise ValueError("El vector personalizado debe ser un diccionario")
                    complete_vector = {key: vector_dict.get(key, 0.0) for key in analyzer.labels}
                    vector = np.array(list(complete_vector.values()), dtype=float)
                except json.JSONDecodeError:
                    raise HTTPException(400, detail="El custom_vector debe ser un JSON válido")
                except ValueError as ve:
                    raise HTTPException(400, detail=str(ve))
            else:
                vector = analyzer.vector(message)
            vector_dict, log = engine.propagate(seed_user, message, max_steps, method=method, custom_vector=vector)
            
            # Calcular alcance final y t_pico
            alcance_final = calculate_alcance_final(log)
            t_pico = calculate_t_pico(log, method="emotion")
            t_max = calculate_t_max(t_pico)
            
            # Calcular nuevas métricas para RIP DSN
            total_nodes = len(engine.graph.nodes()) if engine.graph else 0
            pct_modificar = calculate_pct_modificar(log, total_nodes)
            pct_reenviar = calculate_pct_reenviar(log, total_nodes)
            pct_ignorar = calculate_pct_ignorar(log, total_nodes, pct_reenviar, pct_modificar)
            
            # Save propagation log to MongoDB
            propagation_id = str(uuid.uuid4())
            propagation_document = {
                "propagation_id": propagation_id,
                "propagation_name": propagation_name,
                "seed_user": seed_user,
                "message": message,
                "method": method,
                "tipo_red": tipo_red,  # Usar el valor recibido del frontend
                "metodo": metodo,  # Usar el valor recibido del frontend
                "max_steps": max_steps,
                "thresholds": thresholds_dict,
                "k": k,
                "policy": policy,
                "cluster_filtering": cluster_filtering,
                "alcance_final": alcance_final,
                "t_pico": t_pico,
                "t_max": t_max,
                "pct_modificar": pct_modificar,
                "pct_reenviar": pct_reenviar,
                "pct_ignorar": pct_ignorar,
                "timestamp": datetime.utcnow(),
                "log": log
            }
            try:
                collection.insert_one(propagation_document)
                print(f"Propagation log saved to MongoDB with ID: {propagation_id}")
            except Exception as mongo_error:
                print(f"Error saving to MongoDB: {str(mongo_error)}")
                raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
            
            return {
                "vector": vector_dict,
                "log": log,
                "propagation_id": propagation_id,
                "message": f"Propagación ejecutada correctamente con método {method}",
            }
        elif nodes_csv_file and links_csv_file and not (csv_file or xlsx_file):
            nodes_df = pd.read_csv(nodes_csv_file.file)
            links_df = pd.read_csv(links_csv_file.file)
            simple_engine.build(links_df, nodes_df)
            if seed_user not in simple_engine.nodes:
                raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
            log = simple_engine.propagate(seed_user, message, max_steps)
            
            # Calcular alcance final y t_pico
            alcance_final = calculate_alcance_final(log)
            t_pico = calculate_t_pico(log, method="rip-dsn")
            t_max = calculate_t_max(t_pico)
            
            # Calcular nuevas métricas para RIP DSN
            total_nodes = len(simple_engine.nodes) if simple_engine.nodes else 0
            pct_modificar = calculate_pct_modificar(log, total_nodes)
            pct_reenviar = calculate_pct_reenviar(log, total_nodes)
            pct_ignorar = calculate_pct_ignorar(log, total_nodes, pct_reenviar, pct_modificar)
            
            # Save RIP-DSN propagation log to MongoDB
            propagation_id = str(uuid.uuid4())
            propagation_document = {
                "propagation_id": propagation_id,
                "propagation_name": propagation_name,
                "seed_user": seed_user,
                "message": message,
                "method": "rip-dsn",
                "tipo_red": tipo_red,  # Usar el valor recibido del frontend
                "metodo": metodo,  # Usar el valor recibido del frontend
                "max_steps": max_steps,
                "k": k,
                "policy": policy,
                "cluster_filtering": cluster_filtering,
                "alcance_final": alcance_final,
                "t_pico": t_pico,
                "t_max": t_max,
                "pct_modificar": pct_modificar,
                "pct_reenviar": pct_reenviar,
                "pct_ignorar": pct_ignorar,
                "timestamp": datetime.utcnow(),
                "log": log
            }
            try:
                collection.insert_one(propagation_document)
                print(f"RIP-DSN propagation log saved to MongoDB with ID: {propagation_id}")
            except Exception as mongo_error:
                print(f"Error saving to MongoDB: {str(mongo_error)}")
                raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
            
            return {
                "vector": {},
                "log": log,
                "propagation_id": propagation_id,
                "message": "Propagación RIP-DSN ejecutada correctamente",
            }
        else:
            raise HTTPException(400, detail="Debe proporcionar csv_file+xlsx_file o nodes_csv_file+links_csv_file, pero no ambos.")
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación: {str(e)}")

@app.post("/generate-vectors")
async def generate_vectors(num_vectors: int = Form(..., description="Número de vectores a generar", ge=1, le=1000)):
    """
    Genera el número especificado de vectores sintéticos usando el modelo VAE cargado.
    """
    try:
        df_sintetico = generar_datos_sinteticos_cargado(
            vae_model, scaler, num_vectors, feature_columns, cluster_column
        )
        result = df_sintetico.to_dict(orient='records')
        return {
            "vectors": result,
            "message": f"Se generaron {len(result)} vectores sintéticos correctamente"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al generar vectores: {str(e)}")

@app.post("/propagate-ba-sir")
async def propagate_ba_sir(
    seed_user: str = Form(..., description="Usuario inicial infectado"),
    beta: float = Form(..., description="Tasa de infección", ge=0.0, le=1.0),
    gamma: float = Form(..., description="Tasa de recuperación", ge=0.0, le=1.0),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    nodes_csv_file: UploadFile = File(..., description="CSV con nodos"),
    links_csv_file: UploadFile = File(..., description="CSV con relaciones"),
    max_steps: int = Form(10, ge=1, le=50),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("barabasi-albert", description="Tipo de red"),
    metodo: str = Form("SIR", description="Método de propagación")
):
    """
    Ejecuta propagación SIR (Susceptible-Infected-Recovered) en la red.
    """
    try:
        nodes_df = pd.read_csv(nodes_csv_file.file)
        links_df = pd.read_csv(links_csv_file.file)
        sir_engine.build(links_df, nodes_df)
        
        if seed_user not in sir_engine.nodes:
            raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
        
        log = sir_engine.propagate(seed_user, beta, gamma, max_steps)
        
        # Calcular alcance final y t_pico
        alcance_final = calculate_alcance_final(log)
        t_pico = calculate_t_pico(log, method="sir")
        t_max = calculate_t_max(t_pico)
        
        # Save SIR propagation log to MongoDB
        propagation_id = str(uuid.uuid4())
        propagation_document = {
            "propagation_id": propagation_id,
            "propagation_name": propagation_name,
            "seed_user": seed_user,
            "method": "ba-sir",
            "tipo_red": tipo_red,  # Usar el valor recibido del frontend
            "metodo": metodo,  # Usar el valor recibido del frontend
            "beta": beta,
            "gamma": gamma,
            "k": k,
            "policy": policy,
            "max_steps": max_steps,
            "alcance_final": alcance_final,
            "t_pico": t_pico,
            "t_max": t_max,
            "timestamp": datetime.utcnow(),
            "log": log
        }
        try:
            collection.insert_one(propagation_document)
            print(f"SIR propagation log saved to MongoDB with ID: {propagation_id}")
        except Exception as mongo_error:
            print(f"Error saving to MongoDB: {str(mongo_error)}")
            raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
        
        return {
            "log": log,
            "propagation_id": propagation_id,
            "message": "Propagación SIR ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación SIR: {str(e)}")

@app.post("/propagate-ba-sis")
async def propagate_ba_sis(
    seed_user: str = Form(..., description="Usuario inicial infectado"),
    beta: float = Form(..., description="Tasa de infección", ge=0.0, le=1.0),
    gamma: float = Form(..., description="Tasa de recuperación", ge=0.0, le=1.0),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    nodes_csv_file: UploadFile = File(..., description="CSV con nodos"),
    links_csv_file: UploadFile = File(..., description="CSV con relaciones"),
    max_steps: int = Form(10, ge=1, le=50),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("barabasi-albert", description="Tipo de red"),
    metodo: str = Form("SIS", description="Método de propagación")
):
    """
    Ejecuta propagación SIS (Susceptible-Infected-Susceptible) en la red.
    """
    try:
        nodes_df = pd.read_csv(nodes_csv_file.file)
        links_df = pd.read_csv(links_csv_file.file)
        sis_engine.build(links_df, nodes_df)
        
        if seed_user not in sis_engine.nodes:
            raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
        
        log = sis_engine.propagate(seed_user, beta, gamma, max_steps)
        
        # Calcular alcance final y t_pico
        alcance_final = calculate_alcance_final(log)
        t_pico = calculate_t_pico(log, method="sis")
        t_max = calculate_t_max(t_pico)
        
        # Save SIS propagation log to MongoDB
        propagation_id = str(uuid.uuid4())
        propagation_document = {
            "propagation_id": propagation_id,
            "propagation_name": propagation_name,
            "seed_user": seed_user,
            "method": "ba-sis",
            "tipo_red": tipo_red,  # Usar el valor recibido del frontend
            "metodo": metodo,  # Usar el valor recibido del frontend
            "beta": beta,
            "gamma": gamma,
            "k": k,
            "policy": policy,
            "max_steps": max_steps,
            "alcance_final": alcance_final,
            "t_pico": t_pico,
            "t_max": t_max,
            "timestamp": datetime.utcnow(),
            "log": log
        }
        try:
            collection.insert_one(propagation_document)
            print(f"SIS propagation log saved to MongoDB with ID: {propagation_id}")
        except Exception as mongo_error:
            print(f"Error saving to MongoDB: {str(mongo_error)}")
            raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
        
        return {
            "log": log,
            "propagation_id": propagation_id,
            "message": "Propagación SIS ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación SIS: {str(e)}")

@app.post("/propagate-hk-sir")
async def propagate_hk_sir(
    seed_user: str = Form(..., description="Usuario inicial infectado"),
    beta: float = Form(..., description="Tasa de infección", ge=0.0, le=1.0),
    gamma: float = Form(..., description="Tasa de recuperación", ge=0.0, le=1.0),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    nodes_csv_file: UploadFile = File(..., description="CSV con nodos"),
    links_csv_file: UploadFile = File(..., description="CSV con relaciones"),
    max_steps: int = Form(10, ge=1, le=50),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("holme-kim", description="Tipo de red"),
    metodo: str = Form("SIR", description="Método de propagación")
):
    """
    Ejecuta propagación SIR (Susceptible-Infected-Recovered) en red Holme-Kim.
    """
    try:
        nodes_df = pd.read_csv(nodes_csv_file.file)
        links_df = pd.read_csv(links_csv_file.file)
        sir_engine.build(links_df, nodes_df)
        
        if seed_user not in sir_engine.nodes:
            raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
        
        log = sir_engine.propagate(seed_user, beta, gamma, max_steps)
        
        # Calcular alcance final y t_pico
        alcance_final = calculate_alcance_final(log)
        t_pico = calculate_t_pico(log, method="sir")
        t_max = calculate_t_max(t_pico)
        
        # Save Holme-Kim SIR propagation log to MongoDB
        propagation_id = str(uuid.uuid4())
        propagation_document = {
            "propagation_id": propagation_id,
            "propagation_name": propagation_name,
            "seed_user": seed_user,
            "method": "hk-sir",
            "tipo_red": tipo_red,  # Usar el valor recibido del frontend
            "metodo": metodo,  # Usar el valor recibido del frontend
            "beta": beta,
            "gamma": gamma,
            "k": k,
            "policy": policy,
            "max_steps": max_steps,
            "alcance_final": alcance_final,
            "t_pico": t_pico,
            "t_max": t_max,
            "timestamp": datetime.utcnow(),
            "log": log
        }
        try:
            collection.insert_one(propagation_document)
            print(f"Holme-Kim SIR propagation log saved to MongoDB with ID: {propagation_id}")
        except Exception as mongo_error:
            print(f"Error saving to MongoDB: {str(mongo_error)}")
            raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
        
        return {
            "log": log,
            "propagation_id": propagation_id,
            "message": "Propagación Holme-Kim SIR ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación Holme-Kim SIR: {str(e)}")

@app.post("/propagate-hk-sis")
async def propagate_hk_sis(
    seed_user: str = Form(..., description="Usuario inicial infectado"),
    beta: float = Form(..., description="Tasa de infección", ge=0.0, le=1.0),
    gamma: float = Form(..., description="Tasa de recuperación", ge=0.0, le=1.0),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    nodes_csv_file: UploadFile = File(..., description="CSV con nodos"),
    links_csv_file: UploadFile = File(..., description="CSV con relaciones"),
    max_steps: int = Form(10, ge=1, le=50),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("holme-kim", description="Tipo de red"),
    metodo: str = Form("SIS", description="Método de propagación")
):
    """
    Ejecuta propagación SIS (Susceptible-Infected-Susceptible) en red Holme-Kim.
    """
    try:
        nodes_df = pd.read_csv(nodes_csv_file.file)
        links_df = pd.read_csv(links_csv_file.file)
        sis_engine.build(links_df, nodes_df)
        
        if seed_user not in sis_engine.nodes:
            raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
        
        log = sis_engine.propagate(seed_user, beta, gamma, max_steps)
        
        # Calcular alcance final y t_pico
        alcance_final = calculate_alcance_final(log)
        t_pico = calculate_t_pico(log, method="sis")
        t_max = calculate_t_max(t_pico)
        
        # Save Holme-Kim SIS propagation log to MongoDB
        propagation_id = str(uuid.uuid4())
        propagation_document = {
            "propagation_id": propagation_id,
            "propagation_name": propagation_name,
            "seed_user": seed_user,
            "method": "hk-sis",
            "tipo_red": tipo_red,  # Usar el valor recibido del frontend
            "metodo": metodo,  # Usar el valor recibido del frontend
            "beta": beta,
            "gamma": gamma,
            "k": k,
            "policy": policy,
            "max_steps": max_steps,
            "alcance_final": alcance_final,
            "t_pico": t_pico,
            "t_max": t_max,
            "timestamp": datetime.utcnow(),
            "log": log
        }
        try:
            collection.insert_one(propagation_document)
            print(f"Holme-Kim SIS propagation log saved to MongoDB with ID: {propagation_id}")
        except Exception as mongo_error:
            print(f"Error saving to MongoDB: {str(mongo_error)}")
            raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
        
        return {
            "log": log,
            "propagation_id": propagation_id,
            "message": "Propagación Holme-Kim SIS ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación Holme-Kim SIS: {str(e)}")

@app.post("/propagate-rw-sir")
async def propagate_rw_sir(
    seed_user: str = Form(..., description="Usuario inicial infectado"),
    beta: float = Form(..., description="Tasa de infección", ge=0.0, le=1.0),
    gamma: float = Form(..., description="Tasa de recuperación", ge=0.0, le=1.0),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    nodes_csv_file: UploadFile = File(..., description="CSV con nodos"),
    links_csv_file: UploadFile = File(..., description="CSV con relaciones"),
    max_steps: int = Form(10, ge=1, le=50),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("real-world", description="Tipo de red"),
    metodo: str = Form("SIR", description="Método de propagación")
):
    """
    Ejecuta propagación SIR (Susceptible-Infected-Recovered) en red del mundo real.
    """
    try:
        nodes_df = pd.read_csv(nodes_csv_file.file)
        links_df = pd.read_csv(links_csv_file.file)
        rw_sir_engine.build(links_df, nodes_df)
        
        if seed_user not in rw_sir_engine.nodes:
            raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
        
        log = rw_sir_engine.propagate(seed_user, beta, gamma, max_steps)
        
        # Calcular alcance final y t_pico
        alcance_final = calculate_alcance_final(log)
        t_pico = calculate_t_pico(log, method="sir")
        t_max = calculate_t_max(t_pico)
        
        # Save Real World SIR propagation log to MongoDB
        propagation_id = str(uuid.uuid4())
        propagation_document = {
            "propagation_id": propagation_id,
            "propagation_name": propagation_name,
            "seed_user": seed_user,
            "method": "rw-sir",
            "tipo_red": tipo_red,  # Usar el valor recibido del frontend
            "metodo": metodo,  # Usar el valor recibido del frontend
            "beta": beta,
            "gamma": gamma,
            "k": k,
            "policy": policy,
            "max_steps": max_steps,
            "alcance_final": alcance_final,
            "t_pico": t_pico,
            "t_max": t_max,
            "timestamp": datetime.utcnow(),
            "log": log
        }
        try:
            collection.insert_one(propagation_document)
            print(f"Real World SIR propagation log saved to MongoDB with ID: {propagation_id}")
        except Exception as mongo_error:
            print(f"Error saving to MongoDB: {str(mongo_error)}")
            raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
        
        return {
            "log": log,
            "propagation_id": propagation_id,
            "message": "Propagación Real World SIR ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación Real World SIR: {str(e)}")

@app.post("/propagate-rw-sis")
async def propagate_rw_sis(
    seed_user: str = Form(..., description="Usuario inicial infectado"),
    beta: float = Form(..., description="Tasa de infección", ge=0.0, le=1.0),
    gamma: float = Form(..., description="Tasa de recuperación", ge=0.0, le=1.0),
    k: int = Form(..., description="Valor K", ge=1, le=100),
    policy: str = Form(..., description="Política seleccionada"),
    nodes_csv_file: UploadFile = File(..., description="CSV con nodos"),
    links_csv_file: UploadFile = File(..., description="CSV con relaciones"),
    max_steps: int = Form(10, ge=1, le=50),
    propagation_name: str = Form(..., description="Nombre de la propagación"),
    tipo_red: str = Form("real-world", description="Tipo de red"),
    metodo: str = Form("SIS", description="Método de propagación")
):
    """
    Ejecuta propagación SIS (Susceptible-Infected-Susceptible) en red del mundo real.
    """
    try:
        nodes_df = pd.read_csv(nodes_csv_file.file)
        links_df = pd.read_csv(links_csv_file.file)
        rw_sis_engine.build(links_df, nodes_df)
        
        if seed_user not in rw_sis_engine.nodes:
            raise HTTPException(400, detail=f"El usuario inicial '{seed_user}' no se encuentra en la red")
        
        log = rw_sis_engine.propagate(seed_user, beta, gamma, max_steps)
        
        # Calcular alcance final y t_pico
        alcance_final = calculate_alcance_final(log)
        t_pico = calculate_t_pico(log, method="sis")
        t_max = calculate_t_max(t_pico)
        
        # Save Real World SIS propagation log to MongoDB
        propagation_id = str(uuid.uuid4())
        propagation_document = {
            "propagation_id": propagation_id,
            "propagation_name": propagation_name,
            "seed_user": seed_user,
            "method": "rw-sis",
            "tipo_red": tipo_red,  # Usar el valor recibido del frontend
            "metodo": metodo,  # Usar el valor recibido del frontend
            "beta": beta,
            "gamma": gamma,
            "k": k,
            "policy": policy,
            "max_steps": max_steps,
            "alcance_final": alcance_final,
            "t_pico": t_pico,
            "t_max": t_max,
            "timestamp": datetime.utcnow(),
            "log": log
        }
        try:
            collection.insert_one(propagation_document)
            print(f"Real World SIS propagation log saved to MongoDB with ID: {propagation_id}")
        except Exception as mongo_error:
            print(f"Error saving to MongoDB: {str(mongo_error)}")
            raise HTTPException(500, detail=f"Error saving propagation log to MongoDB: {str(mongo_error)}")
        
        return {
            "log": log,
            "propagation_id": propagation_id,
            "message": "Propagación Real World SIS ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la propagación Real World SIS: {str(e)}")

@app.get("/api/reports")
async def get_reports():
    """
    Obtiene todos los reportes de propagación almacenados en MongoDB.
    """
    try:
        # Obtener todos los documentos de propagación con todos los campos necesarios
        reports = list(collection.find({}, {
            "_id": 1, 
            "propagation_name": 1, 
            "method": 1, 
            "tipo_red": 1, 
            "metodo": 1, 
            "seed_user": 1, 
            "policy": 1, 
            "alcance_final": 1, 
            "t_pico": 1, 
            "t_max": 1,
            "timestamp": 1,
            "beta": 1,
            "gamma": 1,
            "thresholds": 1,
            "pct_modificar": 1,
            "pct_reenviar": 1,
            "pct_ignorar": 1,
            "log": 1,
            "k": 1,
            "max_steps": 1,
            "cluster_filtering": 1
        }))
        
        # Procesar los datos para el frontend
        processed_reports = []
        for report in reports:
            # Usar los campos explícitos si están disponibles, sino usar la lógica de fallback
            network_type = report.get("tipo_red", "unknown")
            propagation_method = report.get("metodo", "unknown")
            
            # Fallback para reportes antiguos sin los nuevos campos
            if network_type == "unknown":
                if "ba-" in report.get("method", ""):
                    network_type = "barabasi-albert"
                elif "hk-" in report.get("method", ""):
                    network_type = "holme-kim"
                elif "rw-" in report.get("method", ""):
                    network_type = "real-world"
                elif report.get("method") == "rip-dsn":
                    network_type = "barabasi-albert"  # RIP-DSN se usa principalmente con BA
                elif report.get("method") in ["ema", "sma"]:
                    network_type = "barabasi-albert"  # Métodos emocionales se usan principalmente con BA
            
            if propagation_method == "unknown":
                method = report.get("method", "")
                if "sir" in method:
                    propagation_method = "SIR"
                elif "sis" in method:
                    propagation_method = "SIS"
                elif method == "rip-dsn":
                    propagation_method = "RIP-DSN"
                elif method in ["ema", "sma"]:
                    propagation_method = "RIP-DSN"  # Métodos emocionales
            
            processed_report = {
                "_id": str(report["_id"]),
                "propagationName": report.get("propagation_name", "Sin nombre"),
                "networkType": network_type,
                "propagationMethod": propagation_method,
                "user": report.get("seed_user", "N/A"),
                "policy": report.get("policy", "N/A"),
                "finalReach": report.get("alcance_final", "N/A"),
                "peakTime": report.get("t_pico", "N/A"),
                "maxPeakTime": report.get("t_max", "N/A"),
                "createdAt": report.get("timestamp", datetime.utcnow()).isoformat(),
                # Campos adicionales para el modal de detalle
                "beta": report.get("beta"),
                "gamma": report.get("gamma"),
                "thresholds": report.get("thresholds"),
                "method": report.get("method"),
                "pct_modificar": report.get("pct_modificar"),
                "pct_reenviar": report.get("pct_reenviar"),
                "pct_ignorar": report.get("pct_ignorar"),
                "log": report.get("log"),
                "k": report.get("k"),
                "max_steps": report.get("max_steps"),
                "cluster_filtering": report.get("cluster_filtering"),
                # Campos originales para compatibilidad
                "propagation_name": report.get("propagation_name", "Sin nombre"),
                "tipo_red": network_type,
                "metodo": propagation_method,
                "seed_user": report.get("seed_user", "N/A"),
                "alcance_final": report.get("alcance_final", "N/A"),
                "t_pico": report.get("t_pico", "N/A"),
                "t_max": report.get("t_max", "N/A")
            }
            processed_reports.append(processed_report)
        
        # Ordenar por fecha de creación (más recientes primero)
        processed_reports.sort(key=lambda x: x["createdAt"], reverse=True)
        
        return processed_reports
        
    except Exception as e:
        print(f"Error fetching reports from MongoDB: {str(e)}")
        raise HTTPException(500, detail=f"Error al obtener los reportes: {str(e)}")

@app.get("/api/test")
async def test_endpoint():
    """
    Endpoint de prueba para verificar conectividad.
    """
    return {"message": "Backend funcionando correctamente", "status": "ok"}

@app.get("/health")
async def health():
    """
    Verifica el estado del servidor.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)