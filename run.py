from src.models import DiamondsPipeline

import pandas as pd

df = pd.read_csv('data/diamonds.csv')
pipeline = DiamondsPipeline(input_data=df, model_name='XGBoostModel')
experiment_id, r2, mae = pipeline.train()
print(experiment_id, r2, mae)