from flask import Flask, render_template, request, redirect, url_for
import pprint
import requests
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
model = pickle.load(open('models/rf_model_dis.pkl','rb'))

def geocode_address(address):
    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': address, 'key': 'AIzaSyC0M3mCvV2JU6u5OkKiTxb4UE5aQttu-lM'}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    
    else:
        print('Error:', response.status_code)
        return False

@app.route('/')
def home():
    return render_template('sign_in.html')

@app.route('/log_in', methods=['GET', 'POST'])
def log_in():
    error = None
    if request.method == 'POST':
        # Replace with your actual verification logic
        if request.form.get('username') == '123@g' and request.form.get('password') == '123':
            return render_template('index2.html')
        else:
            error = 'Invalid credentials'
    return render_template('sign_in.html', error=error)

@app.route('/form_page')
def form_page():
    return render_template('form_page.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = request.form.get('weight')
    location = request.form.get('location')
    symptoms = request.form.get('symptoms')
    duration = request.form.get('duration')
    medical_history = request.form.get('medicalHistory')

    # Now you can use the variables name, age, and gender
    # For example, you can print them to the console

    gmaps_data_dict = geocode_address(location)
    if gmaps_data_dict["status"] == "OK":
        pprint.pp(gmaps_data_dict)
        print()
        results_geometry_bounds = gmaps_data_dict["results"][0]["geometry"]["viewport"]
        northeast = results_geometry_bounds["northeast"]
        southwest = results_geometry_bounds["southwest"]
        lat = (northeast["lat"] + southwest["lat"]) / 2
        lng = (northeast["lng"] + southwest["lng"]) / 2
        coordinates = [lat,lng]
    else:
        coordinates = [None]

#     print(f"""
# Name: {name},
# Age: {age},
# Gender: {gender},
# Height: {height},
# Weight: {weight},
# Location: {location},
# Coordinates: {coordinates},
# Symptoms: {symptoms},
# Duration: {duration},
# Medical History: {medical_history}""")

    patient_pariculars_dict = {
        "name": name,
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "location": location,
        "coordinates": coordinates,
        "symptoms": symptoms,
        "duration": duration,
        "medical_history": medical_history
    }

    pprint.pp(patient_pariculars_dict)

    # Don't forget to return a response or render a template
    return "Form submitted successfully!"

@app.route('/predict',methods=['POST'])
def predict():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = request.form.get('weight')
    location = request.form.get('location')
    symptoms = request.form.get('symptoms')
    duration = request.form.get('duration')
    causes = request.form.get('causes')
    medicine = request.form.get('medicine')
    medical_history = request.form.get('medicalHistory')

    gmaps_data_dict = geocode_address(location)
    if gmaps_data_dict["status"] == "OK":
        pprint.pp(gmaps_data_dict)
        print()
        results_geometry_bounds = gmaps_data_dict["results"][0]["geometry"]["viewport"]
        northeast = results_geometry_bounds["northeast"]
        southwest = results_geometry_bounds["southwest"]
        lat = (northeast["lat"] + southwest["lat"]) / 2
        lng = (northeast["lng"] + southwest["lng"]) / 2
        coordinates = [lat,lng]
    else:
        coordinates = [None]

    # Predicting
    df = pd.read_csv('medical_data.csv')
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'])
    mean_date = df['DateOfBirth'].dropna().mean()
    df['DateOfBirth'].fillna(mean_date, inplace=True)
    columns = ['Gender', 'Symptoms', 'Causes', 'Disease', 'Medicine']
    for column in columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    df['Name'].fillna('Unknown Patient', inplace=True)
    encoder = LabelEncoder()
    encoder.fit(df[['Gender', 'Symptoms', 'Causes', 'Medicine']])
    encoded_array = encoder.transform([gender, symptoms, causes, medicine])
    int_features = list(encoded_array)
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    return render_template('index2.html', prediction_text='Output {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
