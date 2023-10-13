# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
#from sklearn.preprocessing import StandardScaler


# loading the saved model
file_name = 'C:/Users/sanke/OneDrive/Desktop/coding/Python/diabetes_prediction/trained_model.sav'
loaded_model = pickle.load(open(file_name, "rb"))

# Main Program
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

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
    print("The person is not diabetic")
    
else:
    print("The person is diabetic")