from src.lib import TrainingPipeline

import pandas as pd

df = pd.read_csv('data/diamonds.csv')
pipeline = TrainingPipeline(input_data=df.drop(columns=['depth', 'table']))
experiment_id, r2, mae = pipeline.train()
print(experiment_id, r2, mae)