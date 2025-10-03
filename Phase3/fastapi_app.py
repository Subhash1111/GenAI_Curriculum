from fastapi import FastAPI
from pydantic import BaseModel
import mlflow, mlflow.sklearn
import pandas as pd

app = FastAPI(title='Iris Classifier API')

class IrisIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load latest model from mlruns (for demo, we load from local path you saved above)
import os, glob
# Fallback: directly load from the artifact path saved in notebook run
model = mlflow.sklearn.load_model('model')

@app.post('/predict')
def predict(inp: IrisIn):
    df = pd.DataFrame([inp.dict()])
    y = model.predict(df)[0]
    return {'prediction': int(y)}
