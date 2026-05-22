# import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model=joblib.load('auto_price_model.pkl')
model_columns=joblib.load('model_columns.pkl')

explainer=joblib.load('shap_explainer.pkl')

def predict_price(incoming_data_dict):
    df=pd.DataFrame([incoming_data_dict])
    df_encode=pd.get_dummies(df)
    df_onhot=df_encode.reindex(columns=model_columns, fill_value=0)
    prediction=model.predict(df_onhot)
    return round(prediction[0],2)

sample_web_request = {
    'make': 'audi',
    'body-style': 'sedan',
    'engine-size': 136,
    'horsepower': 110,
    'curb-weight': 2844,
    'highway-mpg': 25
}

predicted_price = predict_price(sample_web_request)
print(f"Predicted Price: ${predicted_price}")
