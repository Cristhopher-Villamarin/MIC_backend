# main.py
import nltk
nltk.download("punkt", quiet=True)


from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

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

@app.post("/propagate")
async def propagate(
    seed_user: str = Form(..., description="Usuario origen"),
    message: str = Form(..., description="Mensaje a propagar"),
    csv_file: UploadFile = File(None, description="CSV con aristas"),
    xlsx_file: UploadFile = File(None, description="Excel con estados"),
    nodes_csv_file: UploadFile = File(None, description="CSV con nodos"),
    links_csv_file: UploadFile = File(None, description="CSV con relaciones"),
    max_steps: int = Form(4, ge=1, le=10)
):
    """
    Sube los archivos, construye la red y simula la cascada.
    - Si se proporcionan csv_file y xlsx_file, usa PropagationEngine (con emociones).
    - Si se proporcionan nodes_csv_file y links_csv_file, usa SimplePropagationEngine (sin emociones).
    """
    try:
        if csv_file and xlsx_file and not (nodes_csv_file or links_csv_file):
            # Modo original (con emociones)
            edges_df = pd.read_csv(csv_file.file)
            states_df = pd.read_excel(xlsx_file.file)
            engine.build(edges_df, states_df)  # Red lista
            vector, log = engine.propagate(seed_user, message, max_steps)
            return {
                "vector": vector,
                "log": log,
                "message": "Propagación ejecutada correctamente",
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