import joblib as job
import pandas as pd
from load import*  

import pandas as pd

def Classifier():
    print("===  Classifier ===")

    numeric_features = ['Age', 'Oxygen_saturation', 'Temperature']
    data = {}

    for feature in numeric_features:
        value = float(input(f"Enter {feature}: "))
        data[feature] = value

    categories = {
        'Gender': ['F', 'M'],
        'Cough': ['Bloody', 'Dry', 'Wet'],
        'Shortness_of_breath': ['Mild', 'Moderate', 'Severe'],
        'Chest_pain': ['Mild', 'Moderate', 'Severe'],
        'Fatigue': ['Mild', 'Moderate', 'Severe'],
        'Confusion': ['No', 'Yes'],
        'Sputum_color': ['Bloody', 'Clear', 'Green', 'Yellow']
    }
    for cat, options in categories.items():
        val = input(f"Enter {cat} ({' / '.join(options)}): ").strip().capitalize()
        for option in options:
            col_name = f"{cat}_{option}"
            data[col_name] = 1 if val == option else 0

    ordered_columns = [
        'Age', 'Oxygen_saturation', 'Temperature',
        'Gender_F', 'Gender_M',
        'Cough_Bloody', 'Cough_Dry', 'Cough_Wet',
        'Shortness_of_breath_Mild', 'Shortness_of_breath_Moderate', 'Shortness_of_breath_Severe',
        'Chest_pain_Mild', 'Chest_pain_Moderate', 'Chest_pain_Severe',
        'Fatigue_Mild', 'Fatigue_Moderate', 'Fatigue_Severe',
        'Confusion_No', 'Confusion_Yes',
        'Sputum_color_Bloody', 'Sputum_color_Clear', 'Sputum_color_Green', 'Sputum_color_Yellow'
    ]
    for col in ordered_columns:
        if col not in data:
            
            data[col] = 0

    input_df = pd.DataFrame([data], columns=ordered_columns)

    model = load_model()
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        print("The patient is infected with pneumonia.")
    else:
         print("The patient is not infected with pneumonia.")


