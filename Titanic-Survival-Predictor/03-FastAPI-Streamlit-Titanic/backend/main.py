# ===========================================================
# Import required libraries
# ===========================================================

# FastAPI framework
from fastapi import FastAPI
# Pydantic is used to validate incoming request data
from pydantic import BaseModel, Field
from typing import Literal
# Joblib is used to load our saved machine learning model
import joblib
import pandas as pd

# ===========================================================
# Create FastAPI application
# ===========================================================

app = FastAPI(

    title="Titanic Survival Prediction API",

    description="""
A Machine Learning API built using FastAPI.

This API predicts whether a passenger aboard the Titanic
is likely to survive based on demographic and travel
information.

The trained Scikit-learn Pipeline performs:

- Missing Value Imputation
- One-Hot Encoding
- Feature Scaling
- Logistic Regression Prediction
""",

    version="1.0.0",

    contact={
        "name": "Saleem Shahzad",
        "email": "pmaf16e018@gmail.com"
    }

)

# ===========================================================
# Passenger Schema
# ===========================================================

# This class defines the structure of the JSON
# that our API expects to receive.
#
# FastAPI will automatically:
# • Validate the data
# • Convert JSON into a Python object
# • Display this schema in Swagger UI
#

class Passenger(BaseModel):

    # Passenger Class
    pclass: Literal[1, 2, 3] = Field(
        ...,
        description="Passenger Class (1 = First, 2 = Second, 3 = Third)",
        example=3
    )

    # Passenger Gender
    sex: Literal["male", "female"] = Field(
        ...,
        description="Passenger Gender",
        example="male"
    )

    # Passenger Age
    age: float = Field(
        ...,
        ge=0,
        le=100,
        description="Passenger Age",
        example=25
    )

    # Ticket Fare
    fare: float = Field(
        ...,
        ge=0,
        description="Ticket Fare",
        example=32.5
    )

    # Number of siblings/spouses aboard
    sibsp: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of Siblings/Spouses",
        example=1
    )

    # Number of parents/children aboard
    parch: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of Parents/Children",
        example=0
    )

    # Port of Embarkation
    embarked: Literal["C", "Q", "S"] = Field(
        ...,
        description="Embarkation Port",
        example="S"
    )

    # Travelling Alone
    alone: Literal[0, 1] = Field(
        ...,
        description="Travelling Alone (1 = Yes, 0 = No)",
        example=1
    )

# ===========================================================
# Load the trained model
# ===========================================================

# The model is loaded ONLY ONCE when the server starts.
#
# Every future prediction request will use this same model.
#
# This is much faster than loading model.pkl every time
# someone clicks the Predict button.
#
model = joblib.load("model.pkl")

# ===========================================================
# Home Endpoint
# ===========================================================

@app.get(
    "/",
    summary="API Status",
    description="Returns API health information."
)
def home():

    return {
        "application": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "status": "Running",
        "model_loaded": True
    }

# ===========================================================
# Model Information Endpoint
# ===========================================================

@app.get(
    "/model-info",
    summary="Model Information",
    description="Returns information about the loaded ML model."
)
def model_info():

    return {
        "model_type": str(type(model))
    }


# ===========================================================
# Prediction Endpoint
# ===========================================================

# This endpoint receives passenger information,
# sends it to our trained machine learning pipeline,
# and returns the prediction.

@app.post(
    "/predict",
    summary="Predict Passenger Survival",
    description="""
Predicts whether a Titanic passenger is likely to survive
using the trained Scikit-learn pipeline.
"""
)
def predict(passenger: Passenger):

    # -------------------------------------------------------
    # Convert Passenger object into a Python dictionary
    # -------------------------------------------------------
    #
    # Pydantic stores the incoming JSON as an object.
    # We convert it into a normal dictionary.
    #
    passenger_dict = passenger.model_dump()


    # -------------------------------------------------------
    # Convert dictionary into a Pandas DataFrame
    # -------------------------------------------------------
    #
    # Our scikit-learn Pipeline expects a DataFrame
    # having exactly the same column names that were
    # used during training.
    #
    input_data = pd.DataFrame([passenger_dict])


    # -------------------------------------------------------
    # Make prediction
    # -------------------------------------------------------
    prediction = model.predict(input_data)[0]


    # -------------------------------------------------------
    # Predict probabilities
    # -------------------------------------------------------
    #
    # predict_proba() returns:
    #
    # [
    #   probability_of_class_0,
    #   probability_of_class_1
    # ]
    #
    probabilities = model.predict_proba(input_data)[0]


    # -------------------------------------------------------
    # Convert numerical prediction into a human-readable label
    # -------------------------------------------------------

    prediction_label = (
        "Survived"
        if prediction == 1
        else "Did Not Survive"
    )

    # -------------------------------------------------------
    # Return response
    # -------------------------------------------------------

    return {

        # Indicates whether the request was processed successfully
        "success": True,

        # Prediction details
        "prediction": {

            # Numerical prediction
            "class": int(prediction),

            # Human-readable prediction
            "label": prediction_label,

            # Model confidence
            "confidence": round(float(max(probabilities)), 4)

        },

        # Prediction probabilities
        "probabilities": {

            # Probability of surviving
            "survived": round(float(probabilities[1]), 4),

            # Probability of not surviving
            "did_not_survive": round(float(probabilities[0]), 4)

        }

    }


