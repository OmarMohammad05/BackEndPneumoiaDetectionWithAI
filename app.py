from flask import Flask, request, jsonify
import joblib  
import numpy as np
from load import * 

# Create a API using Python for AI model using Flask freamWork
app = Flask(__name__) # Create API flask,  __name__ to tell  flask this main file, object for app
 
model = load_model() # Open model from load.py file .

# This route app is path to receives data from jsonFile any object to sent.
# To linke URL to specific function.
@app.route('/predict', methods=['POST']) #/predict this path to conect with flutter.
# methods to reccives data from json file.
#  
def predict():
    try:
        data = request.get_json() # To receives data from json file from any places.

        features = [
            data['Age'],
            data['Oxygen_saturation'],
            data['Temperature'],
            data['Gender_F'],
            data['Gender_M'],
            data['Cough_Bloody'],
            data['Cough_Dry'],
            data['Cough_Wet'],
            data['ChestPain_Yes'],
            data['ChestPain_No'],
            data['Vomiting_Yes'],
            data['Vomiting_No'],
            data['Pain_Yes'],
            data['Pain_No'],
        ] # Take a data in the json file and save data in features list 
        # We must retrive data same same name column.

        features_array = np.array([features]) # Convert a data to 2D beacause the model take data 2D.
    
        prediction = model.predict(features_array)[0]  # The model predication data and return the 1st resulte.

        
        return jsonify({'prediction': int(prediction)}) # return label.

    except Exception as e: # No data rececive json file It will return 400 the file is empty or not completed data.
        return jsonify({'error': str(e)}), 400



 