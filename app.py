import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Load Dataset
cus = pd.read_csv("dataset/customer.csv")
order = pd.read_csv("dataset/orders.csv")
df = pd.merge(cus, order, on='CustomerID'  , how='inner')
filtered_df = df.copy()
st.set_page_config(layout="wide")
st.header("Sara Enterprises's Sales Data Analysis...")
st.text("Shape of Data : "+str(df.shape))

# Filters/Slicers
col1,col2,col3,col4,col5 = st.columns(5)
  
with col1:
    cat = st.selectbox("Select Product Category :",["All"]+list(filtered_df['Product Category'].unique()))
    if cat!='All':
        filtered_df = filtered_df[filtered_df["Product Category"]==cat]
with col2:
    scat = st.selectbox("Select Product Sub-Category :",["All"]+list(filtered_df['Product Sub-Category'].unique()))
    if scat!='All':
        filtered_df = filtered_df[filtered_df["Product Sub-Category"]==scat]
with col3:
    state = st.selectbox("Select State :",["All"]+list(filtered_df['State or Province'].unique()))
    if state!='All':
        filtered_df = filtered_df[filtered_df["State or Province"]==state]
with col4:
    city = st.selectbox("Select City :",["All"]+list(filtered_df['City'].unique()))
    if city!='All':
        filtered_df = filtered_df[filtered_df["City"]==city]
with col5:
    gender = st.radio("Select Gender :",['All']+list(filtered_df.Gender.unique()) , horizontal=True)
    if gender!='All':
        filtered_df = filtered_df[filtered_df["Gender"]==gender]

# KPIwith col1:  
col1 , col2 , col3 , col4 , col5 = st.columns(5)
with col1:  
    st.metric("Total Orders", round(filtered_df["Profit"].count(),2) )
with col2:  
    st.metric("Total Sales", round(filtered_df["Amount"].sum(),2) )
with col3:  
    st.metric("Total Profit", round(filtered_df["Profit"].sum(),2) )
with col4:  
    st.metric("Total Sold Quantity", round(filtered_df["Qty"].sum(),2) )
with col5:  
    st.metric("Total Customers", round(len(filtered_df["CustomerID"].unique()),2) )

# Graphs and Plots
col1 , col2 , col3 = st.columns(3)
with col1:
    pbpc = filtered_df.groupby('Product Category')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,4))
    colors = plt.cm.viridis(np.linspace(0,1,len(filtered_df['Product Category'].unique())))
    plt.bar( pbpc['Product Category'] , pbpc['Profit'] , color=colors)
    st.pyplot(fig)

with col2:
    pbpsc = filtered_df.groupby('Product Sub-Category')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,4))
    colors = ['lightgreen' if val>0 else 'red' for val in pbpsc["Profit"]]
    plt.barh( pbpsc['Product Sub-Category'] , pbpsc['Profit'] , color=colors )
    st.pyplot(fig)

with col3:
    pbs = filtered_df.groupby('State or Province')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,4))
    colors = plt.cm.viridis(np.linspace(0,1,len(filtered_df['State or Province'].unique())))
    plt.bar( pbs['State or Province'] , pbs['Profit'] , color=colors)
    st.pyplot(fig)

st.dataframe(filtered_df , height=250)

# Model Load
model = joblib.load("model/model.pkl")

st.subheader("Predict The Profit Inserting Product Category and Sub-Categoey with Sales Amount")
col1,col2,col3 = st.columns(3)
with col1:
    cat = st.selectbox("Select Product Category :",df['Product Category'].unique())
with col2:
    scat = st.selectbox("Select Product Sub-Category :",df['Product Sub-Category'].unique())
with col3:
    amt = st.number_input("Enter Sales Amount")
if st.button("Predict"):
    temp_df = pd.DataFrame({
        'Product Category':[cat],
        'Product Sub-Category':[scat],
        'Amount':[amt]
    })
    predicted_profit = model.predict(temp_df)
    st.success("Predict Profit :  "+str(predicted_profit[0]))