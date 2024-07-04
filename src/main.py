import os

import pandas as pd

from fastapi import FastAPI, HTTPException

from payloads import PredictPayload, SimilarityPayload
from src.models import DiamondsPipeline


base_dir = os.path.dirname(os.path.abspath(__file__))


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


@app.post("/search_diamonds")
def search_diamonds(request: SimilarityPayload):
    """
    Given the features of a diamond, return n samples from the training dataset
    with the same cut, color, and clarity, and the most similar weight.
    """

    df = pd.read_csv(os.path.join(base_dir, "..", "data", "diamonds.csv"))

    filtered_df = df[
        (df['cut'] == request.cut) &
        (df['color'] == request.color) &
        (df['clarity'] == request.clarity)
    ].copy()

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No diamonds found with the specified cut, color, and clarity.")

    filtered_df.loc[:, 'weight_diff'] = abs(filtered_df['carat'] - request.carat)
    similar_diamonds = filtered_df.nsmallest(request.n, 'weight_diff')

    # Convert to dictionary including index
    result = similar_diamonds.drop(columns=['weight_diff']).reset_index().to_dict(orient='records')

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
