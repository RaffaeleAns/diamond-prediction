import os

import pandas as pd

from fastapi import FastAPI, HTTPException

from payloads import PredictPayload, SimilarityPayload
from src.models import DiamondsPipeline


base_dir = os.path.dirname(os.path.abspath(__file__))


app = FastAPI(
    title="Diamond Prediction API",
    description="An API to predict diamond prices and find similar diamonds based on given features.",
    version="1.0.0"
)


@app.post("/predict", summary="Predict Diamond Price", response_description="Predicted price of the diamond")
def predict(request: PredictPayload):
    """
    Predict the price of a diamond based on its features.

    - **model**: The name of the model to use for prediction.
    - **features**: The features of the diamond (carat, cut, color, clarity, etc.).
    """
    features_dict = request.features.model_dump(exclude_none=True)
    df = pd.DataFrame([features_dict])

    pipeline = DiamondsPipeline(input_data=df, model_name=request.model, load_trained_model=True)

    return {'prediction': float(pipeline.predict()[0])}


@app.post("/search_diamonds", summary="Search Similar Diamonds", response_description="List of similar diamonds")
def search_diamonds(request: SimilarityPayload):
    """
    Given the features of a diamond, return n samples from the training dataset
    with the same cut, color, and clarity, and the most similar weight.

    - **carat**: Carat weight of the diamond.
    - **cut**: Cut quality of the diamond.
    - **color**: Color grade of the diamond.
    - **clarity**: Clarity grade of the diamond.
    - **n**: Number of similar diamonds to return.
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
