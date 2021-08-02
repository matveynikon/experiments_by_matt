from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
#we are loading the model using pickle
model = pickle.load(open('model_iris', 'rb'))
def predict_iris(df):
    predictions=model.predict(df)
    df['predictions']=predictions
    return(df)