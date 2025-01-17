from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from components.webcam import generate_video_stream, get_current_caption, get_current_labels

app = FastAPI()

# Configure le répertoire des templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Page d'accueil pour afficher la webcam."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "caption": get_current_caption(),
            "labels": get_current_labels(),
        },
    )

@app.get("/video_feed")
async def video_feed():
    """Génère le flux vidéo pour la webcam."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/caption")
async def caption():
    """Retourne la légende générée pour l'image actuelle."""
    return JSONResponse(content={"caption": get_current_caption()})

@app.get("/labels")
async def labels():
    """Retourne les étiquettes générées pour l'image actuelle."""
    return JSONResponse(content={"labels": get_current_labels()})

# Lancer l'application avec uvicorn
# Commande : uvicorn app:app --reload
