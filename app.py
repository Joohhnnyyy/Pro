import streamlit as st
import pickle
import numpy as np

# Load the trained model and preprocessing objects
model = pickle.load(open("Final_Fertilizer_model.pkl", "rb"))
ct = pickle.load(open("column_transformer.pkl", "rb"))  # Load OneHotEncoder
sc = pickle.load(open("scaler.pkl", "rb"))  # Load StandardScaler

def predict_fertilizer(Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Phosphorous, Potassium):
    # Prepare input data in the same format as training
    input_data = np.array([[Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Phosphorous, Potassium]])

    # Apply OneHotEncoding and Scaling
    transformed_input = ct.transform(input_data)
    scaled_input = sc.transform(transformed_input)

    # Get prediction
    prediction = model.predict(scaled_input)[0]
    
    return prediction

def main():
    st.title("Fertilizer Recommendation System")

    html_temp = """
    <div style="background-color:#025246; padding:10px">
    <h2 style="color:white; text-align:center;">Fertilizer Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # **Numeric Inputs**
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, step=1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
    moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, step=1)
    nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0, max_value=100, step=1)
    phosphorous = st.number_input("Phosphorous (mg/kg)", min_value=0, max_value=100, step=1)
    potassium = st.number_input("Potassium (mg/kg)", min_value=0, max_value=100, step=1)

    # **Dropdown Inputs for Soil Type & Crop Type**
    soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
    crop_type = st.selectbox("Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
                                           "Barley", "Wheat", "Millets", "Oil Seeds", "Pulses", "Ground Nuts"])

    if st.button("Predict"):
        output = predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, phosphorous, potassium)
        st.success(f"Recommended Fertilizer: {output}")

if __name__ == '__main__':
    main()
