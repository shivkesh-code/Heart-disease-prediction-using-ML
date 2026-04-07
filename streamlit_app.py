import streamlit as st
import numpy as np
import pickle
import pandas as pd # Required if your model was fitted with feature names from a DataFrame

# Load the trained model
# Make sure 'heart_disease_model.pkl' is in the same directory as this script
try:
    model_filename = 'heart_disease_model.pkl'
    loaded_model = pickle.load(open(model_filename, 'rb'))
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Please ensure it's in the same directory as 'streamlit_app.py'.")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Heart Disease Prediction")
st.title("Heart Disease Prediction App")
st.write("Enter the patient's medical parameters to predict the likelihood of heart disease.")

# Input fields for the 13 features
st.sidebar.header("Patient Data Input")

# Define the features and their ranges/types
# These ranges are based on the .describe() output from the original notebook
# Adjust these min/max values as per your dataset's actual ranges if different
features_info = {
    'age': {'type': 'slider', 'min': 29, 'max': 77, 'default': 55, 'step': 1},
    'sex': {'type': 'radio', 'options': {0: 'Female', 1: 'Male'}, 'default': 1},
    'cp': {'type': 'slider', 'min': 0, 'max': 3, 'default': 1, 'step': 1, 'help': 'Chest Pain Type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)'},
    'trestbps': {'type': 'slider', 'min': 94, 'max': 200, 'default': 130, 'step': 1},
    'chol': {'type': 'slider', 'min': 126, 'max': 564, 'default': 240, 'step': 1},
    'fbs': {'type': 'radio', 'options': {0: '<= 120 mg/dl', 1: '> 120 mg/dl'}, 'default': 0, 'help': 'Fasting Blood Sugar > 120 mg/dl'},
    'restecg': {'type': 'slider', 'min': 0, 'max': 2, 'default': 1, 'step': 1, 'help': 'Resting Electrocardiographic Results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)'},
    'thalach': {'type': 'slider', 'min': 71, 'max': 202, 'default': 150, 'step': 1},
    'exang': {'type': 'radio', 'options': {0: 'No', 1: 'Yes'}, 'default': 0, 'help': 'Exercise Induced Angina'},
    'oldpeak': {'type': 'slider', 'min': 0.0, 'max': 6.2, 'default': 1.0, 'step': 0.1},
    'slope': {'type': 'slider', 'min': 0, 'max': 2, 'default': 1, 'step': 1, 'help': 'The slope of the peak exercise ST segment'},
    'ca': {'type': 'slider', 'min': 0, 'max': 4, 'default': 0, 'step': 1, 'help': 'Number of major vessels (0-3) colored by flourosopy'},
    'thal': {'type': 'slider', 'min': 0, 'max': 3, 'default': 2, 'step': 1, 'help': 'Thalassemia type (0: NA, 1: normal, 2: fixed defect, 3: reversible defect)'},
}

input_data = []
for feature, info in features_info.items():
    if info['type'] == 'slider':
        value = st.sidebar.slider(f"{feature.replace('_', ' ').title()}", 
                                 info['min'], info['max'], info['default'], info['step'],
                                 help=info.get('help'))
    elif info['type'] == 'radio':
        # Streamlit radio buttons return the value from the options dict
        # We need to ensure we store the integer key (0 or 1)
        display_options = list(info['options'].values())
        selected_display = st.sidebar.radio(f"{feature.replace('_', ' ').title()}", 
                                            display_options,
                                            index=list(info['options'].keys()).index(info['default']),
                                            help=info.get('help'))
        value = next(key for key, val in info['options'].items() if val == selected_display)
    input_data.append(value)

# Convert input_data to a numpy array for prediction
input_data_as_numpy_array = np.asarray(input_data, dtype=object) # Use dtype=object to handle mixed types temporarily

# Ensure all numeric types for model
for i, val in enumerate(input_data_as_numpy_array):
    try:
        input_data_as_numpy_array[i] = float(val) if isinstance(val, (int, float)) else val
    except ValueError:
        pass # Keep original if not convertible directly

# The model expects a 2D array, even for a single instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Create a DataFrame with column names if the model was fitted with pandas DataFrame
# This helps suppress the UserWarning about missing feature names
# Assuming X_OK was the source of feature names, we'll try to recreate them.
# If you don't have X_OK's columns handy, you can manually list them or just ignore the warning.

# The column names from X_OK are:
feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

input_df = pd.DataFrame(input_data_reshaped, columns=feature_columns)


if st.sidebar.button("Predict Heart Disease"):
    prediction = loaded_model.predict(input_df)

    if prediction[0] == 0:
        st.success("**Prediction: The person does NOT have heart disease.**")
    else:
        st.warning("**Prediction: The person HAS heart disease.**")

    st.subheader("Input Data Provided:")
    input_display_df = pd.DataFrame([input_data], columns=feature_columns)
    st.table(input_display_df)

st.sidebar.markdown("""
--- 
*This app uses a Logistic Regression model to predict heart disease.*
""")
