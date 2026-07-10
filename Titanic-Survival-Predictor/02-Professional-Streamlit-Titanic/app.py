import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ---------------------------------------------------------
# 1. Configure the Streamlit page.
# This MUST be the first Streamlit command.
# ---------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Load the trained Scikit-learn pipeline.
# The pipeline already contains preprocessing + model.
# ---------------------------------------------------------
model = joblib.load("Titanic-Survival-Predictor/02-Professional-Streamlit-Titanic/model.pkl")

# ---------------------------------------------------------
# 3. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.write("""
This demo predicts whether a Titanic passenger would
have survived using a Logistic Regression model.

**Dataset:** Seaborn Titanic

**Model:** Logistic Regression

**Deployment:** Streamlit
""")
    st.divider()
    st.write("Developed as part of an AI Engineering course.")

# ---------------------------------------------------------
# 4. Banner + Heading
# ---------------------------------------------------------
st.image("Titanic-Survival-Predictor/02-Professional-Streamlit-Titanic/assets/titanic.jpg", use_container_width=True)

st.title("🚢 Titanic Survival Predictor")

st.write("""
Predict whether a passenger would have survived the
Titanic disaster using a Machine Learning model.
""")

st.divider()

# ---------------------------------------------------------
# 5. Two-column responsive layout
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    sex = st.selectbox(
        "Sex",
        ["male", "female"],
        help="Passenger gender."
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25,
        help="Passenger age."
    )

    embarked = st.selectbox(
        "Embarked",
        ["C", "Q", "S"],
        help="Port where the passenger boarded."
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=25.0,
        help="Ticket fare."
    )

with right:
    pclass = st.slider(
        "Passenger Class",
        1,
        3,
        3,
        help="1 = First, 2 = Second, 3 = Third"
    )

    sibsp = st.slider(
        "Siblings / Spouses",
        0,
        8,
        0
    )

    parch = st.slider(
        "Parents / Children",
        0,
        6,
        0
    )

    alone = st.selectbox(
        "Travelling Alone?",
        ["Yes", "No"]
    )

# ---------------------------------------------------------
# 6. Create DataFrame
# IMPORTANT:
# Column names must match training data.
# ---------------------------------------------------------
input_df = pd.DataFrame({
    "sex":[sex],
    "embarked":[embarked],
    "pclass":[pclass],
    "age":[age],
    "fare":[fare],
    "sibsp":[sibsp],
    "parch":[parch],
    "alone":[1 if alone=="Yes" else 0]
})

st.divider()

# ---------------------------------------------------------
# 7. Prediction Button
# ---------------------------------------------------------
if st.button("🔍 Predict Survival", use_container_width=True):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    survival_probability = probability[1]

    st.header("Prediction Result")

    if prediction == 1:
        st.success("✅ Passenger is likely to SURVIVE.")
    else:
        st.error("❌ Passenger is unlikely to SURVIVE.")

    st.metric(
        "Survival Probability",
        f"{survival_probability:.2%}"
    )

    st.progress(float(survival_probability))

    if survival_probability >= 0.80:
        st.info("Confidence: High")
    elif survival_probability >= 0.60:
        st.info("Confidence: Moderate")
    else:
        st.info("Confidence: Low")

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Feature":[
            "Sex","Age","Passenger Class",
            "Fare","Embarked","Siblings/Spouses",
            "Parents/Children","Travelling Alone"
        ],
        "Value":[
            sex,age,pclass,
            fare,embarked,sibsp,
            parch,alone
        ]
    })

    st.table(summary)

    with st.expander("📄 View Input Data"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("🤖 Model Information"):
        st.write("""
**Algorithm:** Logistic Regression

Accuracy: **83.80%**

Precision: **77.59%**

Recall: **73.77%**

F1 Score: **75.63%**
""")

    st.caption(
        f"Prediction generated on "
        f"{datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}"
    )

st.divider()
st.caption("Built with ❤️ using Streamlit and Scikit-learn.")
