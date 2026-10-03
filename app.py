import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='Insurance Cost Prediction',
    page_icon='logo/logo.jpg',
    layout='centered'
)

model_path = Path('outputs/best_insurance_model.joblib')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error('Model file not found')
    st.info(
        "Please make sure 'best_insurance_model.joblib' "
        "exists inside the models folder."
    )
    st.stop()

st.title("Medical Insurance Cost Predictor")

st.write(
    "Enter the customer's information below to predict "
    "their estimated medical insurance charges."
)

st.divider()

with st.form('insurance_form'):
    st.subheader('Customer information')

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30,
            step=1
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
    with col2:
        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            [
                "southwest",
                "southeast",
                "northwest",
                "northeast"
            ]
        )
    predict_button = st.form_submit_button(
        "Predict Insurance Cost"
    )

if predict_button:

    # Create DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.divider()

    st.subheader("Prediction Result")

    st.success(
        f"Estimated Insurance Cost: ${prediction:,.2f}"
    )

    # Show input data
    with st.expander("View Input Data"):
        st.dataframe(
            input_data,
            use_container_width=True
        )