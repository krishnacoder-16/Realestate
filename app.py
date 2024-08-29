import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import numpy as np
data1 = pd.read_csv("bengaluru_house_prices.csv")
data2= pd.read_csv("processed_data.csv")
x = pd.read_csv("x.csv")
with open('banglore_home_prices_model.pickle','rb') as f:
   model= pickle.load(f)
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns==location)[0][0]
    X = np.zeros(len(x.columns))
    X[0]= sqft
    X[1]=bath
    X[2]=bhk
    if loc_index>=0:
        X[loc_index]= 1
    return model.predict([X])[0]

st.title("Real Estate Price Predictor for Banglore")
nav = st.sidebar.radio("Navigation",["Home","Prediction"])
if nav=="Home":
    if nav=="Home":
        img = Image.open("reimage.jpg")
        st.image(img, width=800)
    if st.checkbox("Show Dataframe Used to Train the Model"):
        st.dataframe(data1)
        st.download_button(
   "Press to Download",
   data1.to_csv(index=False).encode("utf-8"),
   "estate.csv",
   "text/csv",
   key='download-csv'
)
if nav=="Prediction":
    column = []
    for i in (data2.columns):
          column.append(i)
    column.append("Other")
    loc = st.selectbox("Choose Your Location", column[4:])
    BHK= st.slider("Enter BHK", 1,16,1)
    bath= st.number_input("Enter Number of bathrooms", min_value=1, max_value=16)
    area = st.number_input("Enter Area in Square Ft.", min_value=50, max_value=25000)
    if st.button("Predict"):
        price = predict_price(loc,area,bath,BHK)
        st.write("## Rs.(Lcs) ",price )
