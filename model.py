import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Random forest for regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import  mean_squared_error, mean_absolute_error,r2_score  # Metrics to use for evaluation
from sklearn.model_selection import  train_test_split # Split arrays or matrices into random train and test subsets.
cars = pd.read_csv('co2_emissions_canada.csv')
cars.rename(columns = lambda x : x.lower().replace(" ", "_"),inplace = True)
cars['co2_emissions(g/km)'] = cars['co2_emissions(g/km)'].astype(float)
cars.drop(columns= ['make', 'model', 'transmission'], inplace = True)
cars.drop_duplicates(inplace= True)
ohe = OneHotEncoder(handle_unknown = 'ignore')

# Fit and transform the data
transformed = ohe.fit_transform(cars[['fuel_type']])

# Make a copy of the data to work on
cars_ml = cars
cars_ml[ohe.categories_[0]] = transformed.toarray()
X = cars_ml.drop(['vehicle_class', 'fuel_type', 'co2_emissions(g/km)'], axis = 1)
y = cars_ml['co2_emissions(g/km)']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

st.header('Co2 emissions prediction')
st.caption('The data comes from DataCamp competition: Everyone can learn python.')
st.caption('the purpose of this app is to use the interactive UI to make prediction about carbon emissions and to help comapanies understand how to make more friendly vehicles')
st.markdown('Link to my [EDA and Model](https://app.datacamp.com/workspace/w/f885974b-5e59-4465-b2a1-4377297b0e50) on DataCamp')
with st.sidebar:
    slider_engine = st.number_input( label = 'engine_size(l)')

    slider_cylinder = st.number_input( label = 'cylinders')

    slider_fuel = st.number_input(label = 'fuel_consumption_comb_(l/100_km)')

    dropdown = st.selectbox(options=['E', 'Z', 'D','X','N'], label='fuel_type')

# creat a dict

def create_feature_matrix(slider_engine, slider_cylinder,slider_fuel, dropdown):
    data = {}
    data['engine_size(l)'] = slider_engine
    data['cylinders'] = slider_cylinder
    data['fuel_type'] = dropdown
    data['fuel_consumption_comb_(l/100_km)'] = slider_fuel
    df = pd.DataFrame(data, index=[0])
    ohe_1 = OneHotEncoder(handle_unknown = 'ignore')
    ohe_1.fit(cars[['fuel_type']])
    transformed_1 = ohe_1.transform(df[['fuel_type']])
    df[ohe_1.categories_[0]] = transformed_1.toarray()
    df.drop(['fuel_type'], axis = 1, inplace=True)
    return df




def predict_carbon():
    feature_matrix = create_feature_matrix(slider_engine, slider_cylinder,slider_fuel, dropdown)
    prediction = rf.predict(feature_matrix)
    st.text(f"Estimated Co2 emissions is : {prediction:} (g/km)")

button = st.button(label="Predict carbon emissions",on_click=predict_carbon)
