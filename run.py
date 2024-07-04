from src.models import TrainingPipeline

import pandas as pd

df = pd.read_csv('data/diamonds.csv')
pipeline = TrainingPipeline(input_data=df, model_name='LinearRegressionModel')
experiment_id, r2, mae = pipeline.train()
print(experiment_id, r2, mae)