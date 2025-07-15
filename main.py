import nltk
nltk.download("punkt", quiet=True)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from generate_vectors import generar_datos_sinteticos_cargado, cargar_modelo_y_escalador
from utils import EmotionAnalyzer, PropagationEngine, SimplePropagationEngine

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
    custom_vector: str = Form(None, description="JSON con vector emocional personalizado")
):
    """
    Sube los archivos, construye la red y simula la cascada.
    - Si se proporcionan csv_file y xlsx_file, usa PropagationEngine (con emociones).
    - Si se proporcionan nodes_csv_file y links_csv_file, usa SimplePropagationEngine (sin emociones).
    """
    try:
        thresholds_dict = json.loads(thresholds) if thresholds else {}
        if csv_file and xlsx_file and not (nodes_csv_file or links_csv_file):
            if method not in ["ema", "sma"]:
                raise HTTPException(400, detail="El método debe ser 'ema' o 'sma'")
            edges_df = pd.read_csv(csv_file.file)
            states_df = pd.read_excel(xlsx_file.file)
            engine.build(edges_df, states_df, thresholds=thresholds_dict)
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
            return {
                "vector": vector_dict,
                "log": log,
                "message": f"Propagación ejecutada correctamente con método {method}",
            }
        elif nodes_csv_file and links_csv_file and not (csv_file or xlsx_file):
            nodes_df = pd.read_csv(nodes_csv_file.file)
            links_df = pd.read_csv(links_csv_file.file)
            simple_engine.build(links_df, nodes_df)
            log = simple_engine.propagate(seed_user, message, max_steps)
            return {
                "vector": {},
                "log": log,
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

@app.get("/health")
async def health():
    """
    Verifica el estado del servidor.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)