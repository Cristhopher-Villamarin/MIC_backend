# main.py
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from utils import EmotionAnalyzer, PropagationEngine

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

analyzer = EmotionAnalyzer()        #  ⇠  /analyze
engine   = PropagationEngine()      #  ⇠  /propagate

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
    seed_user: str         = Form(..., description="Usuario origen"),
    message:   str         = Form(..., description="Mensaje a propagar"),
    csv_file:  UploadFile  = File(..., description="CSV con aristas"),
    xlsx_file: UploadFile  = File(..., description="Excel con estados"),
    max_steps: int         = Form(4, ge=1, le=10)
):
    """
    Sube los archivos, construye la red y simula la cascada.
    """
    try:
        edges_df  = pd.read_csv(csv_file.file)
        states_df = pd.read_excel(xlsx_file.file)
        engine.build(edges_df, states_df)               # red lista

        vector, log = engine.propagate(seed_user, message, max_steps)
        return {
            "vector": vector,
            "log":    log,
            "message": "Propagación ejecutada correctamente",
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
