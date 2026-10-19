# Mushroom Classifier

A small FastAPI app that predicts whether a mushroom is edible or poisonous using a trained Random Forest model.

## Run the API

Install the dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the API.

## Endpoint

### `POST /predict`

Send the five mushroom feature codes as JSON:

```json
{
  "odor": "a",
  "gill-size": "b",
  "gill-color": "k",
  "stalk-surface-above-ring": "s",
  "spore-print-color": "n"
}
```

The response contains a prediction: `edible` or `poisonous`. Use feature codes found in the training data.

## Model

The model uses odor, gill size, gill color, stalk surface above the ring, and spore-print color. It is a tuned Random Forest with 50 trees and a maximum depth of 10.

**Safety:** Do not use this model to decide whether a real mushroom is safe to eat.
