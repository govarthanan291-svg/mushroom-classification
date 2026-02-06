import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained Gradient Boosting Classifier model
with open("mushroom_model (1).pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit page config
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="centered"
)

# Title of the app
st.markdown("<h1 style='text-align:center;'>🍄 Mushroom Classification App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Predict whether a mushroom is edible or poisonous!</h4>", unsafe_allow_html=True)

# User input fields
st.subheader("📋 Enter Mushroom Characteristics")

cap_shape = st.selectbox("Cap Shape", ["bell", "conical", "flat", "knobbed", "sunken"])
cap_surface = st.selectbox("Cap Surface", ["fibrous", "grooves", "scaly", "smooth"])
cap_color = st.selectbox("Cap Color", ["brown", "buff", "cinnamon", "gray", "green", "pink", "purple", "red", "white", "yellow"])

# Add all the other relevant features based on your dataset columns
# Example:
odor = st.selectbox("Odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])

# Add more features from the dataset

# Encoding the input data (same preprocessing as the training data)
input_data = pd.DataFrame({
    "cap_shape": [cap_shape],
    "cap_surface": [cap_surface],
    "cap_color": [cap_color],
    "odor": [odor],
    # Add other features accordingly
})

# Convert categorical variables to numerical values (one-hot encoding for the input data)
input_data_encoded = pd.get_dummies(input_data)

# Make the prediction
if st.button("🔮 Predict Mushroom Class"):
    # Ensure the input features match the training dataset's format (same number of columns)
    prediction = model.predict(input_data_encoded)

    if prediction[0] == 'p':
        st.error("❌ The mushroom is **poisonous**!")
    else:
        st.success("✅ The mushroom is **edible**!")
