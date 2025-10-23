import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load model and encoder
@st.cache_resource
def load_model_and_encoder():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model_and_encoder()

# Predefined categories (based on notebook uniques; adjust if needed)
type_options = ['BHK1', 'BHK2', 'BHK3', 'BHK4', 'BHK4PLUS', 'RK1', 'PENTHOUSE']
lease_type_options = ['ANYONE', 'FAMILY', 'BACHELOR', 'COMPANY']
furnishing_options = ['SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED']
parking_options = ['BOTH', 'BIKE', 'CAR', 'NONE']
facing_options = ['E', 'NE', 'N', 'SE', 'NW', 'S', 'W', 'SW']
water_supply_options = ['CORP_BORE', 'CORPORATION', 'BOREWELL']
building_type_options = ['AP', 'IF', 'IH', 'GC']

# Amenities list
amenities_list = ['LIFT', 'GYM', 'INTERNET', 'AC', 'CLUB', 'INTERCOM', 'POOL', 'CPA', 'FS', 'SERVANT', 'SECURITY', 'SC', 'GP', 'PARK', 'RWH', 'STP', 'HK', 'PB', 'VP']

# Streamlit app
st.title("Rental Property Price Predictor")

# Form for inputs
with st.form(key='rental_form'):
    # Date
    activation_date = st.date_input("Activation Date", value=datetime.today())
    
    # Location (lat/long)
    latitude = st.number_input("Latitude", value=12.93, format="%.6f")
    longitude = st.number_input("Longitude", value=77.67, format="%.6f")
    
    # Categorical
    type_ = st.selectbox("Type", type_options)
    lease_type = st.selectbox("Lease Type", lease_type_options)
    furnishing = st.selectbox("Furnishing", furnishing_options)
    parking = st.selectbox("Parking", parking_options)
    facing = st.selectbox("Facing", facing_options)
    water_supply = st.selectbox("Water Supply", water_supply_options)
    building_type = st.selectbox("Building Type", building_type_options)
    
    # Numerical/Bool
    property_size = st.number_input("Property Size (sq ft)", min_value=100, value=1400)
    property_age = st.number_input("Property Age (years)", min_value=0, value=4)
    bathroom = st.number_input("Bathrooms", min_value=1, value=2)
    cup_board = st.number_input("Cupboards", min_value=0, value=2)
    floor = st.number_input("Floor", min_value=0, value=3)
    total_floor = st.number_input("Total Floors", min_value=1, value=4)
    balconies = st.number_input("Balconies", min_value=0, value=2)
    negotiable = st.checkbox("Negotiable", value=True)
    
    # Amenities checkboxes
    st.subheader("Amenities")
    amenities_dict = {}
    for amen in amenities_list:
        amenities_dict[amen] = 1 if st.checkbox(amen, value=False) else 0
    
    submit = st.form_submit_button("Predict Rental Price")

if submit:
    # --- Extract date parts ---
    year = activation_date.year
    month = activation_date.month
    day = activation_date.day
    
    # --- Create user dictionary ---
    user_dict = {
        'month': month,
        'day': day,
        'year': year,
        'latitude': latitude,
        'longitude': longitude,
        'type': type_,
        'lease_type': lease_type,
        'furnishing': furnishing,
        'parking': parking,
        'facing': facing,
        'water_supply': water_supply,
        'building_type': building_type,
        'property_size': property_size,
        'property_age': property_age,
        'bathroom': bathroom,
        'cup_board': cup_board,
        'floor': floor,
        'total_floor': total_floor,
        'balconies': balconies,
        'negotiable': 1 if negotiable else 0,
        # Redundant for backward compatibility
        'gym': amenities_dict['GYM'],
        'lift': amenities_dict['LIFT'],
        'swimming_pool': amenities_dict['POOL']
    }
    
    # --- Convert to DataFrame ---
    user_df = pd.DataFrame([user_dict])
    
    # --- Ensure categorical columns are strings ---
    cat_cols = ['type', 'lease_type', 'furnishing', 'parking', 'facing', 'water_supply', 'building_type']
    user_df[cat_cols] = user_df[cat_cols].astype(str)
    
    # --- Encode categoricals ---
    try:
        encoded_array = encoder.transform(user_df[cat_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))
    except Exception as e:
        st.error(f"Error encoding categorical features: {e}")
        st.stop()
    
    # --- Drop original categorical columns and add encoded ones ---
    user_df = pd.concat([user_df.drop(columns=cat_cols), encoded_df], axis=1)
    
    # --- Add all amenities (ensure all expected amenity columns exist) ---
    amenities_df = pd.DataFrame([amenities_dict])
    # If the model expects more/different amenities, you may need to add missing columns
    user_df = pd.concat([user_df, amenities_df], axis=1)
    
    # --- Reorder columns to match training if needed ---
    try:
        user_df = user_df[model.feature_names_in_]
    except:
        st.warning("Column order mismatch with model. Make sure the feature columns match training.")
    
    # --- Predict ---
    try:
        prediction = model.predict(user_df)
        st.success(f"Predicted Rental Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")



