import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.express as px

# Load model
model = joblib.load("temperature_model.pkl")

# Load dataset (for averages)
df = pd.read_csv("pakistan_weather_2000_2024.csv")

# Convert date
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

st.title("🌍 Pakistan Temperature Prediction Map")

# --- Map Data ---
map_data = df[['city', 'latitude', 'longitude']].drop_duplicates()

fig = px.scatter_mapbox(
    map_data,
    lat="latitude",
    lon="longitude",
    hover_name="city",
    zoom=4,
    height=500
)

fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)

# --- User Input ---
st.subheader("📍 Select Location")

lat = st.number_input("Latitude", value=30.0)
lon = st.number_input("Longitude", value=70.0)

# Current month
month = datetime.datetime.now().month

# --- Smart Feature Generation ---
def get_features(lat, lon, month):
    # nearest data
    temp_df = df.copy()
    
    # Filter by month
    temp_df = temp_df[temp_df['month'] == month]
    
    # Compute averages
    avg_values = temp_df.mean(numeric_only=True)
    
    features = avg_values.to_dict()
    
    # Override location
    features['latitude'] = lat
    features['longitude'] = lon
    features['month'] = month
    features['dayofweek'] = datetime.datetime.now().weekday()
    
    return pd.DataFrame([features])

# --- Prediction ---
if st.button("Predict Temperature"):
    input_data = get_features(lat, lon, month)
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"🌡️ Predicted Temperature: {prediction:.2f} °C")