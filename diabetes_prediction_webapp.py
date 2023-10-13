# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:32:52 2023

@author: sanke
"""

import numpy as np
import pickle
import streamlit as st
import os

# loading the saved model
current_directory = os.path.dirname(os.path.realpath(__file__))
relative_file_path = "trained_model.sav"
file_name = os.path.join(current_directory, relative_file_path)
loaded_model = pickle.load(open(file_name, "rb"))

def diabetes_prediction(input_data):

    # changing the input_data into numpy array
    input_data_convert = np.asarray(input_data)

    # reshaping the array
    input_data_reshaped = input_data_convert.reshape(1, -1)


    ## standardize the input data
    #scaler = StandardScaler()
    #scaler.fit(input_data_reshaped)
    #input_data_final = scaler.transform(input_data_reshaped)

    print(input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    print("Prediction = ", prediction)

    if (prediction[0] == 0 ):
        return "The person is not diabetic"
        
    else:
        return "The person is diabetic"
        

def main():
    # title
    st.title("Diabetes Prediction Webapp")
    
    
    # inputs
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    
    Pregnancies = st.text_input("No. of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    Skinthickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")
    
    
    # Prediction Value
    diagnosis = ""
    
    # submission
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([
            Pregnancies,
            Glucose,
            BloodPressure,
            Skinthickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age])
    
    st.success(diagnosis)
    
    
if __name__ == "__main__":
    main()