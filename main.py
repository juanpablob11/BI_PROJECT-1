# main.py
from fastapi import FastAPI, Request
from typing import Optional
from DataModel import DataModel
from PredictionModel import Model
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

# Servir archivos estáticos como CSS desde /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Carpeta para los HTMLs
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_classify(request: Request):
    return templates.TemplateResponse("classify.html", {"request": request})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    
    model = Model(df.columns)
    result = model.make_predictions(df)
    return {"prediccion": float(result[0])}

