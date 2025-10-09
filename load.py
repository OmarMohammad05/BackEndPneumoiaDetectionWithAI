import joblib
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def traningModel(data):
    X = data.drop(["Diagnosis"], axis=1)
    y = data["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    
    y_test_pred = model.predict(X_test)

 
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_test_pred))

    joblib.dump(model, "pneumoniaModel.pkl")
    print("Model saved as 'pneumonia_model.pkl'")

    return model

########### ####################  ##################  #######################

        
def load_model():
    return joblib.load("pneumoniaModel.pkl")



    
    
    
    
    