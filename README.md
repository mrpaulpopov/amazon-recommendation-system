## Overview
This project implements a recommendation system for predicting user preferences based on historical interactions.
The model combines embedding representations with matrix factorization and is evaluated on feedback data.

## Features
- Matrix factorization with biases
- GPU/CPU training
- Dockerized pipeline
- Class 'Config' with constants and paths

## How to use
1. Download the dataset with ```/download``` (~876 MB)
2. Run ```/preprocess```
3. Train the model ```/train```
4. Run inference ```/predict```. 

The model predicts the probability that the user will like each item.

## Building
Two profiles are available for running the project:

**GPU (with CUDA support):**

``` docker compose --profile gpu up --build```

**CPU only:**

``` docker compose --profile cpu up --build```

### Dataset Overview
The dataset contains user-item interactions with ratings.

| item               | user                  | rating | timestamp      | user_id | item_id |
|--------------------|-----------------------|--------|----------------|---------|---------|
| B000K8PH8C         | A3PHJ4NMHMBBUB        | 5.0    | 1391212800     | 0       | 0       |
| B001T6BK6M         | A3DTVMQGMNLX26        | 2.0    | 1392854400     | 1       | 1       |
| B007GFX0PY         | A2ZGNB9CWL7SLK        | 1.0    | 1437091200     | 2       | 2       |

Prediction is computed as:

r̂(u, i) = μ + b_u + b_i + <p_u, q_i>

## Pipeline
- downloading dataset
- preprocessing
- training
- inference

### Data Preprocessing
The preprocessing pipeline (implemented with Pandas) includes:
1. Removing users and items with fewer than 5 interactions.
2. Sampling the dataset using ```.sample()``` (configurable parameter).
3. Encoding users and items with ```.factorize()```.
4. Saving mappings (user - id, item - id, and vice versa).

### Training
Prerequisites: processed dataset file, mapping file.

Training: uses a DataLoader. Each epoch iterates over: user_batch, item_batch, rating_batch.


The model is based on matrix factorization:
- user and item embeddings are learned
- Bias terms are included for users and items


RMSE >> MAE → есть выбросы
RMSE ≈ MAE → ошибки равномерные
У тебя:
RMSE ≈ 1.18
MAE ≈ 0.90
MAE 0.9 → в среднем ты ошибаешься почти на 1 рейтинг
RMSE выше → иногда ошибаешься сильно (например 2–3 балла)
У меня регрессионная модель! и это рекомендательная система!

Matrix factorization for explicit feedback!
