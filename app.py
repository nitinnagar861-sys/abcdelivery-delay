import streamlit as st
import joblib
import pandas as pd

st.title('Delivery Delay Prediction')

# Load the trained model
# Assuming the model is saved at '/content/delivery_delay.sav'
model = joblib.load(open('/content/delivery_delay.sav', 'rb'))

# Define the expected feature names from X.columns
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

# Create input widgets for each feature
with st.sidebar:
    st.header('Input Features')
    inputs = {}
    for feature in feature_names:
        # You might want to customize the input type (slider, selectbox, etc.)
        # based on the nature of each feature (e.g., integer, float, categorical)
        if feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score']:
            inputs[feature] = st.slider(f'Enter {feature.replace("_", " ")}', min_value=1, max_value=10, value=5, step=1)
        elif feature in ['Driver_Experience', 'Warehouse_Processing_Time']:
            inputs[feature] = st.slider(f'Enter {feature.replace("_", " ")}', min_value=1, max_value=100, value=50, step=1)
        else:
            inputs[feature] = st.number_input(f'Enter {feature.replace("_", " ")}', value=10.0, step=0.1)

# When the 'Predict' button is clicked
if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the inputs
    input_df = pd.DataFrame([inputs])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1] # Probability of delay

    st.subheader('Prediction Results')
    if prediction[0] == 1:
        st.error(f'Delivery is predicted to be **Delayed** (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Delivery is predicted to be **On Time** (Probability: {1 - prediction_proba[0]:.2f})')

    st.write('---')
    st.subheader('Input Data Used for Prediction:')
    st.write(input_df)
