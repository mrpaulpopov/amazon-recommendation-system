# Amazon Recommendation System

## Overview
This project implements a recommendation system trained on real Amazon review data.
The model predicts user-item preference scores using matrix factorization with bias terms and embedding representations.
This is a regression over explicit feedback (ratings).

**Pipeline**: Data Download - Preprocessing - Training - Evaluation - API (Inference)  
**Tech Stack**: Python, PyTorch, FastAPI, Pandas, Docker, TensorBoard

## Features
- Docker-based pipeline (GPU/CPU support)
- Data preprocessing with Pandas
- API with FastAPI
- Input validation using Pydantic
- API key protection via FastAPI Depends
- Configuration via Config class (paths, parameters)



## ML Features
- Matrix factorization with user and item embeddings (with bias terms)
- Batch training via DataLoader
- TensorBoard logging
- Manual learning rate scheduling based on epoch (currently disabled)
- Train/validation split
- Saving model to the file + JSON metadata storage
- Metrics: MSE, RMSE, MAE

## Limitations
- Popular items have a greater score than the known user-item pairs due to global bias effects.
- Model outputs scores (not probabilities).
- Inference currently uses Pandas DataFrame, which might limit performance.
- Currently, there are more modern recommendation systems in the world.

## Usage
After starting the services:
- API documentation: http://localhost:8000/docs
- TensorBoard logs: http://localhost:6006/
1. Download dataset: ```/download``` (~876 MB)
2. Run preprocessing: ```/preprocess```
3. Train model: ```/train```
4. Run inference: ```/predict``` or ```/topk```. 

## Output
The model produces a relative preference score indicating how likely a user is to interact positively with an item.
Additionally, it supports Top-K recommendation output.

## Docker Setup
Two execution profiles are available:

**GPU (CUDA-enabled):**

``` docker compose --profile gpu up --build```

**CPU-only:**

``` docker compose --profile cpu up --build```

## Dataset
The dataset contains explicit user-item ratings from Amazon reviews.

| item               | user                  | rating | timestamp      |
|--------------------|-----------------------|--------|----------------|
| B000K8PH8C         | A3PHJ4NMHMBBUB        | 5.0    | 1391212800     |
| B001T6BK6M         | A3DTVMQGMNLX26        | 2.0    | 1392854400     |
| B007GFX0PY         | A2ZGNB9CWL7SLK        | 1.0    | 1437091200     |

## Model Formulation
$r̂(u, i) = μ + b_u + b_i + <p_u, q_i> $  
μ — global mean rating  
bᵤ — user bias  
bᵢ — item bias  
pᵤ, qᵢ — latent embeddings  

## Data Preprocessing
Implemented using Pandas:
1. Filter users and items with fewer than 5 interactions
2. Optional dataset subsampling via ```.sample()```
3. Encode users and items using ```.factorize()```
4. Save mappings (user - id, item - id, and reverse mappings for inference).

## Training Details
Training uses batch optimization over **user-item-rating** triplets.
Model learns: user embeddings, item embeddings, bias parameters.

## Results
> **Epoch 70/70 | train_loss=0.0455 | val_loss=0.0855 | val_rmse=0.2924 | val_mae=0.2158**

### Interpretation
Using TensorBoard plots,
- No clear overfitting observed.
- 20 epochs are sufficient for given hyperparameters.
- Error levels are stable across validation set (MAE ≈ 0.22, RMSE ≈ 0.29)

| ![train_loss.png](docs/plots/train_loss.png) | ![val_loss.png](docs/plots/val_loss.png) |
|----------------------------------------------|------------------------------------------|
| ![val_rmse.png](docs/plots/val_rmse.png)     | ![val_mae.png](docs/plots/val_mae.png)   |


### Key Achievements
- Built a full end-to-end recommendation system pipeline from raw Amazon review data to production-ready API.
- Implemented a scalable matrix factorization model with bias terms and embedding representations for explicit feedback prediction.
- Designed a reproducible ML workflow with Docker (CPU/GPU support), enabling consistent training and inference environments.
- Integrated FastAPI inference service with authentication, validation (Pydantic), and batch prediction endpoints.
- Implemented experiment tracking via TensorBoard for monitoring training dynamics and model convergence.