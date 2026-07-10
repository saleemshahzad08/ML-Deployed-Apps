
# ==========================================================
# Titanic Survival Prediction App
#
# This Streamlit application acts as the FRONTEND.
#
# It:
# • Collects passenger information
# • Sends it to a FastAPI backend
# • Receives the prediction
# • Displays the result to the user
#
# The Machine Learning model is loaded ONLY by FastAPI.
# ==========================================================

# Import Streamlit for building the web application.
import streamlit as st

import requests

# ==========================================================
# FastAPI Backend URL
# ==========================================================

API_URL = "https://titanic-survival-api-oisr.onrender.com"

# ----------------------------------------------------------
# App Title
# ----------------------------------------------------------

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter the passenger details below and click **Predict** "
    "to estimate whether the passenger would survive."
)

# ----------------------------------------------------------
# User Inputs
# ----------------------------------------------------------

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

embarked = st.selectbox(
    "Embarked Port",
    ["C", "Q", "S"]
)

pclass = st.slider(
    "Passenger Class",
    min_value=1,
    max_value=3,
    value=3
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

sibsp = st.slider(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.slider(
    "Number of Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

alone = st.selectbox(
    "Travelling Alone?",
    ["Yes", "No"]
)

# ----------------------------------------------------------
# Convert "Yes" / "No" into integers.
#
# During training, the "alone" column was converted into:
#
# True  -> 1
# False -> 0
# ----------------------------------------------------------

alone = 1 if alone == "Yes" else 0



# ----------------------------------------------------------
# Predict Button
# ----------------------------------------------------------

if st.button("Predict"):

    # ------------------------------------------------------
    # Create a dictionary containing all passenger details.
    #
    # This dictionary matches the Passenger schema defined
    # in our FastAPI backend.
    # ------------------------------------------------------

    passenger_data = {

        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare,
        "sibsp": sibsp,
        "parch": parch,
        "embarked": embarked,
        "alone": alone

    }

    # ------------------------------------------------------
    # Display the JSON that will be sent to the API.
    #
    # This is useful for learning and debugging.
    # ------------------------------------------------------

    st.subheader("Request Sent to FastAPI")

    st.json(passenger_data)

    # ------------------------------------------------------
    # Send the HTTP POST request to FastAPI.
    #
    # The 'json=' parameter automatically converts the
    # Python dictionary into JSON before sending it.
    # ------------------------------------------------------

    try:
        with st.spinner("Predicting..."):
            response = requests.post(

                API_URL,

                json=passenger_data,

                timeout=10

            )

        # Raise an exception if the API returned an error.
        response.raise_for_status()

        # Convert JSON response into a Python dictionary.
        result = response.json()

    # ------------------------------------------------------
    # Handle the case where FastAPI is not running.
    # ------------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI server.\n\n"
            "Please make sure the backend is running."
        )

        st.stop()

    # ------------------------------------------------------
    # Handle any other request-related errors.
    # ------------------------------------------------------

    except requests.exceptions.RequestException as e:

        st.error(f"❌ API Error:\n\n{e}")

        st.stop()

    # ------------------------------------------------------
    # Extract values from the API response.
    # ------------------------------------------------------

    prediction_label = result["prediction"]["label"]

    confidence = result["prediction"]["confidence"]

    survival_probability = result["probabilities"]["survived"]

    death_probability = result["probabilities"]["did_not_survive"]

    # ------------------------------------------------------
    # Display prediction.
    # ------------------------------------------------------

    st.subheader("Prediction Result")

    if prediction_label == "Survived":

        st.success(
            f"🎉 Prediction: {prediction_label}"
        )

    else:

        st.error(
            f"❌ Prediction: {prediction_label}"
        )

    # ------------------------------------------------------
    # Display confidence and probabilities.
    # ------------------------------------------------------

    st.write(f"**Model Confidence:** {confidence:.2%}")

    st.write(
        f"**Survival Probability:** "
        f"{survival_probability:.2%}"
    )

    st.write(
        f"**Probability of Not Surviving:** "
        f"{death_probability:.2%}"
    )
