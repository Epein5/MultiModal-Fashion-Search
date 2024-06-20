from fastapi import FastAPI, Form
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from classs import *

app = FastAPI()
app.mount("/datasets", StaticFiles(directory="../Datasets/44k/Images"), name="datasets")
app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")

match_finder = MatchFinder(
    model_path=CFG.model_path,
    embeddings_path=CFG.embeddings_path,
)

class PromptRequest(BaseModel):
    prompt: str


@app.get("/")
async def serve_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/find_matches")
# async def find_matches_endpoint(request: PromptRequest, n: int = 9):
#     prompt = request.prompt
#     matches = match_finder.find_matches(query=prompt, n=n)
#     matches = [f"/datasets/{match}" for match in matches]
#     # return {"matches": matches}
#     return templates.TemplateResponse("matches.html", {"request": request, "matches": matches})

@app.post("/find_matches")
async def find_matches_endpoint(request: Request, prompt: str = Form(...), n: int = 8):
    matches = match_finder.find_matches(query=prompt, n=n)
    matches = [f"/datasets/{match}" for match in matches]
    return templates.TemplateResponse("matches.html", {"request": request, "matches": matches})
