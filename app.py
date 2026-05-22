import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


@st.cache_resource
def load_artifacts():
    model=joblib.load('auto_price_v2_model.pkl')
    model_columns=joblib.load('v2_model_columns.pkl')
    explainer=joblib.load('shap_explainer_v2.pkl')
    return model, model_columns, explainer

rf_v2, model_columns, explainer_v2 = load_artifacts()

st.title("🚗Automobile Price Prediction")
st.write("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox(
        "Car Make", 
        ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 
         'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury', 
         'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 
         'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
    )
    city_mpg = st.number_input("City MPG", min_value=10, max_value=60, value=25)
    highway_mpg = st.number_input("Highway MPG", min_value=10, max_value=60, value=30)
    engine_size = st.number_input("Engine Size (cc)", min_value=50, max_value=400, value=120)

with col2:
    curb_weight = st.number_input("Curb Weight (lbs)", min_value=1000, max_value=5000, value=2500)
    width = st.number_input("Car Width (inches)", min_value=50.0, max_value=80.0, value=65.0)
    wheel_base = st.number_input("Wheel Base (inches)", min_value=80.0, max_value=130.0, value=98.0)



if st.button("Predict Price"):
    user_data = {
        'make': make,
        'city-mpg': city_mpg,
        'highway-mpg': highway_mpg,
        'engine-size': engine_size,
        'curb-weight': curb_weight,
        'width': width,
        'wheel-base': wheel_base
        }

df_input = pd.DataFrame([user_data])

cat_cols = ['make']
df_encoded = pd.get_dummies(df_input, columns=cat_cols)

df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)

prediction = rf_v2.predict(df_aligned)[0]

st.success(f"### Estimated Market Value: ${prediction:,.2f}")

st.balloons()