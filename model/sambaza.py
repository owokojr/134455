import streamlit as st
import pandas as pd
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
X_train = class_data.iloc[:, :-1]  # Features
y_train = class_data.iloc[:, -1]   # Target

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

user_input = np.array([[lead_time, availability, num_products_sold, revenue_generated]])

# Predict supplier reliability based on user inputs
threshold = 10  # Set a threshold for the margin of values

if class_data.empty or 'Costs' not in class_data.columns:
    st.write("Error: 'Costs' column not found or class_data is empty.")
else:
    if lead_time > 15 or availability > 51 or revenue_generated > 19580.0:
        reliability_result = 'Unreliable'
    else:
        reliability_result = 'Reliable'

    # Button to display the classification
    if st.button('Classify Supplier Reliability'):
        st.write(f"### Supplier Reliability Prediction:")
        st.write(f"The supplier is predicted to be: **{reliability_result}**")



