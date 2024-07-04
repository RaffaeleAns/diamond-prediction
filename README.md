# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

# How to run
### Prerequisites
Before running the pipeline, ensure you have the following:

- Python installed (version 3.12)
- Required Python packages installed (found in requirements.txt)

You can install the required packages using the following command:

```bash
pip install requirements.txt
```

## Getting started
### Step 1: Prepare your dataset

Ensure your dataset include the following columns:

- carat: float
- cut: str
- color: str
- clarity: str
- depth: float (mandatory only for XGBoost)
- table: float (mandatory only for XGBoost)
- x: float
- y: float
- z: float
- price: int

if any unexpected column is found, the pipeline will raise an error.
Also, ensure the columns have the correct type.

### Step 2: Run the Training Pipeline
You can run the training pipeline using the following Python code:

```python
import pandas as pd
from src.models import DiamondsPipeline

# Load the dataset
df = pd.read_csv('yourfile.csv')

# Initialize the training pipeline
pipeline = DiamondsPipeline(input_data=df, model_name='XGBoostModel')

# Train the model
experiment_id, r2, mae = pipeline.train()

print(f"Experiment ID: {experiment_id}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
```
You can choose between XGBoostModel and LinearRegressionModel.


Alternatively, you can run the run.py script which uses the data/diamonds.csv file:

```bash
python run.py
```

For each run, the results are stored in experiments/experiments_tracking.json, while the model is saved in a folder named with the uuid of the experiment.

Following a log record of an experiment:

```json
{
    "uuid": "05a9ec3b-3552-421c-92ec-2017f6b95b6e",
    "timestamp": "2024-06-29T22:26:14.730766",
    "date": "2024-06-29 22:26",
    "model_name": "LinearRegressionModel",
    "model_params": {
        "copy_X": true,
        "fit_intercept": true,
        "n_jobs": null,
        "positive": false
    },
    "results": {
        "r2": 0.8074,
        "mae": 511257.98
    }
}
```

### Step 3: Run the API and Obtain Predictions and Similar Diamonds

If you want to predict the value of a new diamond or retrieve the most similar ones already in the database, you can use the API. You can run it using the following command at the project directory:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Once the application startup is complete, you can interact with the API using various methods. The easiest way to explore and test the API endpoints is through the automatically generated Swagger UI.

#### Accessing the Swagger UI

1. **Open Your Browser**: Once the server is running, open your web browser.
2. **Navigate to the Swagger UI**: Go to `http://localhost:8000/docs`.

The Swagger UI (`http://localhost:8000/docs`) provides a graphical interface where you can interact with the API. You can see all available endpoints, their request parameters, and try them out directly from the browser.

#### API Endpoints

The API provides the following main endpoints:

1. **Predict the Price of a Diamond**: `/predict`
2. **Search for Similar Diamonds**: `/search_diamonds`

#### Endpoint Details

##### 1. Predict the Price of a Diamond

- **Endpoint**: `/predict`
- **Method**: POST
- **Description**: Predict the price of a diamond based on its features.

**Request Payload**:
```json
{
  "model": "linear_regression", 
  "features": {
    "carat": 0.23,
    "cut": "Ideal",
    "color": "E",
    "clarity": "SI2",
    "depth": null,
    "table": null,
    "x": 3.95,
    "y": 3.98,
    "z": 2.43
  }
}
```

**Example Using Swagger**:
1. In the Swagger UI, locate the `/predict` endpoint.
2. Click on the endpoint to expand it.
3. Click on the "Try it out" button.
4. Enter the required details in the request payload.
5. Click on the "Execute" button to send the request and see the response.

**Response**:
- **Success**:
  ```json
  {
    "prediction": 1500.0  
  }
  ```
- **Error**: If there are any issues with the request, you will receive an appropriate HTTP status code and error message.

##### 2. Search for Similar Diamonds

- **Endpoint**: `/search_diamonds`
- **Method**: POST
- **Description**: Given the features of a diamond, return `n` samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

**Request Payload**:
```json
{
  "carat": 0.23,
  "cut": "Ideal",
  "color": "E",
  "clarity": "SI2",
  "n": 5
}
```

**Example Using Swagger**:
1. In the Swagger UI, locate the `/search_diamonds` endpoint.
2. Click on the endpoint to expand it.
3. Click on the "Try it out" button.
4. Enter the required details in the request payload.
5. Click on the "Execute" button to send the request and see the response.

**Response**:
- **Success**:
  ```json
  [
    {
      "index": 5,
      "carat": 0.23,
      "cut": "Ideal",
      "color": "E",
      "clarity": "SI2",
      "depth": 61.5,
      "table": 55.0,
      "x": 3.95,
      "y": 3.98,
      "z": 2.43
    },
  ]
  ```
- **Error**: If there are no matching diamonds, or if there are any issues with the request, you will receive an appropriate HTTP status code and error message.
