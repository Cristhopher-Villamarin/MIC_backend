# main.py
import nltk
nltk.download("punkt", quiet=True)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np

from utils import EmotionAnalyzer, PropagationEngine, SimplePropagationEngine

app = FastAPI(
    title="Backend · Propagación Emocional",
    description="Endpoints de prueba para propagar mensajes en una red",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

analyzer = EmotionAnalyzer()               # ⇠ /analyze
engine = PropagationEngine()              # ⇠ /propagate (original)
simple_engine = SimplePropagationEngine()  # ⇠ /propagate (RIP-DSN)

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
        raise HTTPException(500, str(e))

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
                # Complete missing keys with 0
                complete_vector = {key: vector.get(key, 0.0) for key in analyzer.labels}
                return {
                    "vector": complete_vector,
                    "message": "Vector personalizado recibido correctamente",
                }
            except json.JSONDecodeError:
                raise HTTPException(400, "El custom_vector debe ser un JSON válido")
            except ValueError as ve:
                raise HTTPException(400, str(ve))
        return {
            "vector": analyzer.as_dict(message),
            "message": "Mensaje analizado correctamente",
        }
    except Exception as e:
        raise HTTPException(500, str(e))

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
            # Modo original (con emociones)
            if method not in ["ema", "sma"]:
                raise HTTPException(400, "El método debe ser 'ema' o 'sma'")
            edges_df = pd.read_csv(csv_file.file)
            states_df = pd.read_excel(xlsx_file.file)
            engine.build(edges_df, states_df, thresholds=thresholds_dict)  # Red lista
            # Use custom vector if provided, otherwise analyze the message
            if custom_vector:
                try:
                    vector_dict = json.loads(custom_vector)
                    if not isinstance(vector_dict, dict):
                        raise ValueError("El vector personalizado debe ser un diccionario")
                    # Complete missing keys with 0
                    complete_vector = {key: vector_dict.get(key, 0.0) for key in analyzer.labels}
                    # Convert dictionary to NumPy array for PropagationEngine
                    vector = np.array(list(complete_vector.values()), dtype=float)
                except json.JSONDecodeError:
                    raise HTTPException(400, "El custom_vector debe ser un JSON válido")
                except ValueError as ve:
                    raise HTTPException(400, str(ve))
            else:
                vector = analyzer.vector(message)
            # Call propagate with custom_vector as NumPy array
            vector_dict, log = engine.propagate(seed_user, message, max_steps, method=method, custom_vector=vector)
            return {
                "vector": vector_dict,  # Already a dictionary from PropagationEngine
                "log": log,
                "message": f"Propagación ejecutada correctamente con método {method}",
            }
        elif nodes_csv_file and links_csv_file and not (csv_file or xlsx_file):
            # Modo RIP-DSN (sin emociones)
            nodes_df = pd.read_csv(nodes_csv_file.file)
            links_df = pd.read_csv(links_csv_file.file)
            simple_engine.build(links_df, nodes_df)  # Red lista
            log = simple_engine.propagate(seed_user, message, max_steps)
            return {
                "vector": {},  # Sin vector emocional
                "log": log,
                "message": "Propagación RIP-DSN ejecutada correctamente",
            }
        else:
            raise HTTPException(400, "Debe proporcionar csv_file+xlsx_file o nodes_csv_file+links_csv_file, pero no ambos.")
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)