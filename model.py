import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Random forest for regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import  mean_squared_error, mean_absolute_error,r2_score  # Metrics to use for evaluation
from sklearn.model_selection import  train_test_split # Split arrays or matrices into random train and test subsets.
rf = pickle.load(open('final_model.sav', 'rb'))
cars = pd.read_csv('co2_emissions_canada.csv')
cars.rename(columns = lambda x : x.lower().replace(" ", "_"),inplace = True)
cars['co2_emissions(g/km)'] = cars['co2_emissions(g/km)'].astype(float)
cars.drop_duplicates(inplace= True)

st.set_page_config(layout="wide")
st.title('Canadian vehicles analytics')
st.caption('The data comes from DataCamp competition: Everyone can learn python.')
st.caption('the purpose of this app is to use the interactive UI to make prediction about carbon emissions and to help comapanies understand how to make more friendly vehicles')
st.markdown('Link to my [EDA and Model](https://app.datacamp.com/workspace/w/f885974b-5e59-4465-b2a1-4377297b0e50) on DataCamp')





def make_bar(data):
    st.bar_chart(data)

def make_area(data):
    st.area_chart(data)
tab1, tab2, tab3 = st.tabs(["Average fuel consumption", "Average Co2 emissions", "personlised data on your choice of car"])
with tab1:
    st.header('Compare fuel consumption among various companies')

    vehicle_filter = st.selectbox("Choose the car type you would like to know about", pd.unique(cars["vehicle_class"]),key = "fuel")


    cars_dash1 = cars[cars['vehicle_class'] == vehicle_filter].groupby('make')['fuel_consumption_comb_(l/100_km)'].mean().sort_values(ascending = False)
    cars_dash2 = cars[cars['vehicle_class'] == vehicle_filter].groupby('make')['co2_emissions(g/km)'].mean().sort_values(ascending = False)

    make_bar(cars_dash1)
with tab2:
    st.header('Compare Co2 emissions among various companies')
    vehicle_filter_2 = st.selectbox("Choose the car type you would like to know about", pd.unique(cars["vehicle_class"]), key = "carbon")
    cars_dash2_2 = cars[cars['vehicle_class'] == vehicle_filter_2].groupby('make')['co2_emissions(g/km)'].mean().sort_values(ascending = False)
    make_area(cars_dash2_2)
with tab3:
    st.header('Is you car environment friendly?')
    st.subheader('Will you pay extra taxes for your fuel consumption?')
    vehicle_filter_3 = st.selectbox("Choose the car type you would like to know about", pd.unique(cars["vehicle_class"]),key = "tab_3_class")
    df_4=cars[cars['vehicle_class'] == vehicle_filter_3].groupby('make')['co2_emissions(g/km)'].mean()

    company_filter = st.selectbox("Filter by company",options= list(df_4.index),key = "tab_3_co")
    value_emi = round(df_4[company_filter],2)

    df_5=cars[cars['vehicle_class'] == vehicle_filter_3].groupby('make')['fuel_consumption_comb_(l/100_km)'].mean()
    value_fuel = round(df_5[company_filter],2)
    col1,col2 = st.columns(2)
    col1.metric(label = 'Avg Co2 emissions',value= f"{value_emi} g/km")
    col2.metric(label = 'Avg fuel consumption',value= f"{value_fuel} l/100 km",delta = value_fuel - 13, delta_color = 'inverse')
    with col2:
        tax = pd.read_csv('tax.csv')
        st.markdown(':green[Green] means no taxes 🎉')
        st.markdown(':red[Red] means you will pay taxes, refer to the below table ')
        st.table(tax)
    with col1:
        st.markdown(':green[Go green to save Earth from warming.]🌍')
        st.markdown(":green[It's time to cut carbon!]🌱")
        green = pd.read_csv('green.csv')
        st.table(green)
   
    with st.container():
        st.markdown(f"#### Here's a list of the fuel efficient models from {company_filter}   {vehicle_filter_3}")
        suggest = cars[(cars['make'] == company_filter) & (cars['vehicle_class'] == vehicle_filter_3) & (cars['fuel_consumption_comb_(l/100_km)'] < 13.0) ]
        suggest = suggest.drop(columns = ['make', 'vehicle_class'])
        suggest = suggest.drop_duplicates(subset=['model'])
        st.table(suggest)



with st.sidebar:
    st.header("Co2 prediction based on your consumption. check it now!")
    slider_engine = st.slider(label = 'engine_size(l)', min_value=0.0, max_value=10.0, step=0.1)
    
    slider_cylinder = st.slider( label = 'cylinders', min_value=0, max_value=16, step=1 )

    slider_fuel = st.number_input(label = 'fuel_consumption_comb_(l/100_km)')
    
    dropdown = st.selectbox(options=['E', 'Z', 'D','X','N'], label='fuel_type')
    st.markdown('* D: Diesel')
    st.markdown('* E: Ethanol')
    st.markdown('* X: Regular gasoline')
    st.markdown('* N: Natural gas')
    st.markdown('* Z: Premium gasoline')        




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
    if prediction <= 100:
        st.markdown(f"Estimated Co2 emissions is : :green[{prediction:} (g/km)]")
    elif prediction <= 150:
        st.markdown(f"Estimated Co2 emissions is : :blue[{prediction:} (g/km)]")
    elif prediction > 150 and prediction <= 255:
        st.markdown(f"Estimated Co2 emissions is : :orange[{prediction:} (g/km)]")
    elif prediction > 255:
        st.markdown(f"Estimated Co2 emissions is : :red[{prediction:} (g/km)]")
with st.sidebar:        
    button = st.button(label="Predict carbon emissions",on_click=predict_carbon)
