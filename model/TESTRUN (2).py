#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the supply_chain dataset
data = pd.read_csv("supply_chain.csv")

# Defining classes based on specified rows
classes = {
    'Highly Reliable': 4,
    'Unreliable': 16,
    'Moderately Reliable': 22
}

# Extracting specific rows for classes
class_data = data.iloc[[4, 16, 22], :][['Lead time', 'Availability', 'Number of products sold', 'Revenue generated', 'Costs']]

# Training data for the model
X_train = class_data.iloc[:, 16, 4, 5, 6, 24]  # Features
y_train = class_data.iloc[:, 14]   # Target

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Streamlit app layout
st.title('Supplier Reliability Classifier')

# User inputs for specific columns
st.sidebar.header('User Inputs')
lead_time = st.sidebar.number_input('Enter Lead Time', min_value=0)
availability = st.sidebar.number_input('Enter Availability', min_value=0.0)
num_products_sold = st.sidebar.number_input('Enter Number of Products Sold', min_value=0)
revenue_generated = st.sidebar.number_input('Enter Revenue Generated', min_value=0.0)
costs = st.sidebar.number_input('Enter Costs', min_value=0.0)

# Predict supplier reliability based on user inputs
user_input = np.array([[lead_time, availability, num_products_sold, revenue_generated]])

lr_prediction = lr_model.predict(user_input)

# Mapping predictions to the specified classes
for class_name, row_index in classes.items():
    if np.array_equal(lr_prediction, class_data.iloc[row_index, :24]):
        reliability_result = class_name
        break
    else:
        reliability_result = 'Unknown'

st.write(f"### Supplier Reliability Prediction:")
st.write(f"The supplier is predicted to be: **{reliability_result}**")

