from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from IMDB_Project.pipeline import prediction_pipeline as pipe
from IMDB_Project.pipeline import training_pipeline as train_pipe
from IMDB_Project.logger import log


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "name": "HomePage"})


@app.post("/predict", response_class=HTMLResponse)
async def input_prediction(request: Request, review: str = Form(...)):
    result = pipe.prediction_pipeline(review)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


@app.get("/train", response_class=HTMLResponse)
async def train_model(request: Request):
    try:
        log.info("----- Training Pipeline Started -----")
        train_pipe.training_pipeline()  # Run your model training here
        log.info("----- Training Pipeline Completed Successfully -----")
        message = "✅ Model training completed successfully!"
    except Exception as e:
        log.error(f"Training failed: {str(e)}")
        message = f"❌ Training failed: {str(e)}"

    return templates.TemplateResponse("model_training.html", {
        "request": request,
        "message": message
    })