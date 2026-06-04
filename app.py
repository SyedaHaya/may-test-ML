import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import pickle

import plotly.express as px 
st.set_page_config(layout="wide")

select = option_menu(
    menu_title=None,
    options=[
        "Home","Price Prediction","Location Analysis","Data Explorer","About Project" ],
    icons=[
        "house","currency-dollar","geo-alt","door-open","bounding-box","table","info-circle"],orientation="horizontal")
df=pd.read_csv("house_price_prediction_data.csv")
if select == "Home":

    st.title("🏠 House Price Prediction Dashboard")

    st.write("""
    Welcome to the House Price Prediction App.
    This application predicts house prices based on:
    - Location
    - Area (SQ-FT)
    - Number of Rooms
    """)

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Houses", len(df))

    with col2:
        st.metric("Locations", df["location"].nunique())

    with col3:
        st.metric("Average Price", round(df["price_pkr"].mean()))

if select == "Price Prediction":
    st.title("🏠 House Price Prediction")
    le = LabelEncoder()
    df["location"] = le.fit_transform(df["location"])
    X = df[["area_sqft", "rooms", "location"]]  # independent variables
    y = df["price_pkr"]                         # dependent variable

    model = linear_model.LinearRegression()
    model.fit(X, y)
    col9, col10 = st.columns(2)
    with col9:
        area = st.number_input("Enter Area (SQ-FT)", min_value=500)

    with col10:
        rooms = st.number_input("Enter Rooms", min_value=1)

    location = st.selectbox("Select Location",le.classes_)

    if st.button("Predict Price"):

        location_encoded = le.transform([location])[0]

        input_data = [[area, rooms, location_encoded]]

        prediction = model.predict(input_data)
        st.balloons()

        st.success(
            f"Predicted House Price: Rs. {prediction[0]}"
        )
if select == "Location Analysis":

    st.title("📍 Location Analysis")
    df2 = df.groupby("location")["price_pkr"].mean().reset_index()
    st.dataframe(df2)     

    fig = px.bar(
    df2,
    x="location",
    y="price_pkr",
    color="location",
    title="Average House Price by Location")

    st.plotly_chart(fig, use_container_width=True)

    avg_price = df.groupby("location")["price_pkr"].mean().reset_index()

    fig1 = px.line(
        avg_price,
        x="location",
        y="price_pkr",
        markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)


if select == "Data Explorer":

    st.title("📊 Data Explorer")

    st.subheader("Complete Dataset")
    st.dataframe(df)

    location = st.multiselect("Select Location",df["location"].unique())

    st.dataframe(df)

    st.subheader("Area vs Price")

    fig = px.scatter(
        df,
        x="area_sqft",
        y="price_pkr",
        color="location")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("House Distribution by Location")

    

    fig2 = px.pie(
        df,
        names="location",
        values="price_pkr",
        color="location",
        title="Price Distribution by Location"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Average Price by Location")


    fig3 = px.bar(
        df,
        x="location",
        y="price_pkr",
        color="location",
        title="Average Price by Location"
    )

    st.plotly_chart(fig3, use_container_width=True)







    

    
if select == "About Project":

    st.title("ℹ️ About This Project")

    st.write("""
    This is a **House Price Prediction Web App** built using Machine Learning and Streamlit.

    ### 🎯 Objective:
    The main goal of this project is to predict house prices based on:
    - Location
    - Area (Square Feet)
    - Number of Rooms

    ### 🧠 Machine Learning Model:
    We used **Linear Regression** model for prediction.

    ### 📊 Dataset:
    The dataset contains information about houses such as:
    - Location
    - Area
    - Rooms
    - Price

    🛠️ Tools & Technologies:
    - Python 🐍
    - Pandas
    - Scikit-learn
    - Streamlit
    - Plotly

    🚀 Features:
    - Price Prediction
    - Data Exploration
    - Location Analysis
    - Interactive Charts

    👨‍💻 Developer:
    This project is developed for learning and practice purpose in Machine Learning and Data Science.
    """)
    st.success("This project helps users estimate house prices quickly using AI.")
    
    



