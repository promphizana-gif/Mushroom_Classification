from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Initialize FastAPI App
app = FastAPI(
    title="Mushroom Classifier API",
    description="Production API using an optimized Random Forest Model trained on the Top 5 categorical features."
)

# Load production artifacts exported from the notebook
model = joblib.load("mushroom_model.pkl")
encoders = joblib.load("mushroom_encoders.pkl")

# Define JSON request format schema
class MushroomInput(BaseModel):
    odor: str = Field(description="Odor categorical letter code")
    gill_size: str = Field(alias="gill-size", description="Gill size categorical letter code")
    gill_color: str = Field(alias="gill-color", description="Gill color categorical letter code")
    stalk_surface_above_ring: str = Field(alias="stalk-surface-above-ring", description="Stalk surface above ring code")
    spore_print_color: str = Field(alias="spore-print-color", description="Spore print color code")

    class Config:
        populate_by_name = True  # Allows parsing raw JSON containing hyphens seamlessly

@app.post("/predict")
def predict_mushroom(data: MushroomInput):
    try:
        # Organize client payload data to match notebook training order
        input_data = {
            'odor': data.odor, 
            'gill-size': data.gill_size,
            'gill-color': data.gill_color,
            'stalk-surface-above-ring': data.stalk_surface_above_ring,
            'spore-print-color': data.spore_print_color
        }
        
        # Apply Saved LabelEncoders dynamically to clean incoming text strings
        encoded_vector = []
        for col, val in input_data.items():
            encoded_val = encoders[col].transform([val])[0]
            encoded_vector.append(encoded_val)
            
        # Run calculation through the tuned model structure
        prediction = model.predict([encoded_vector])[0]
        verdict = "poisonous" if prediction == 1 else "edible"
        
        return {
            "status": "success",
            "prediction": verdict
        }
        
    except ValueError as encoder_err:
        return {
            "status": "error", 
            "message": f"Invalid code value passed. Ensure entries use true Kaggle string representations. Details: {str(encoder_err)}"
        }
    except Exception as e:
        return {"status": "error", "message": f"An unhandled operational error occurred: {str(e)}"}
