from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from payloads import PredictPayload
from src.models import DiamondsPipeline


app = FastAPI()


@app.post("/predict")
def predict(request: PredictPayload):
    """
    Predict the price of a diamond based on its features.
    """
    features_dict = request.features.model_dump(exclude_none=True)
    df = pd.DataFrame([features_dict])

    pipeline = DiamondsPipeline(input_data=df, model_name=request.model, load_trained_model=True)

    return {'prediction': float(pipeline.predict()[0])}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
