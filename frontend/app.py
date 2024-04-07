import sys
import os
import glob
import re
import pprint
import requests
import pickle
import configparser
import psycopg2
import numpy as np
import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
from joblib import load

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

print(sklearn.__version__)
app = Flask(__name__)
app.config['EXPLAIN_TEMPLATE_LOADING'] = True

config = configparser.ConfigParser()
config.read("dev.config")
config_details = config["DEFAULT"]


model1 = joblib.load(open('rf_model_dis.joblib','rb'))

clf = joblib.load(open('vectormodels/rf_model_dis_vector.joblib','rb'))
# with open('vectormodels/rf_clf.pkl', 'rb') as file:
#     clf = pickle.load(file)
loaded_vec = joblib.load(open('vectormodels/vectorizer.joblib','rb'))

# model = pickle.load(open('models/rf_model_dis.joblib','rb'))
# model = load('models/rf_model_dis.joblib')
# img_model = load_model(MODEL_PATH)
# img_model._make_predict_function()

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
    
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/')
def home():
    return render_template('sign_in.html')

@app.route('/log_in', methods=['GET', 'POST'])
def log_in():
    error = None
    if request.method == 'POST':
        # Replace with your actual verification logic
        if request.form.get('username') == '123@g' and request.form.get('password') == '123':
            return render_template('index1.html')
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
    try:
        if (type(gmaps_data_dict) == dict) and (gmaps_data_dict["status"] == "OK"):
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
    except Exception:
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

@app.route('/predict1',methods=['GET', 'POST'])
def predict1():
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

    if gender == "male":
        gender = "Male"
    else:
        gender = "Female"

    gmaps_data_dict = geocode_address(location)
    try:
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
            coordinates = [0,0]
    except TypeError:
        coordinates = [0,0]

    # Predicting 1
    df = pd.read_csv('medical_data.csv')
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'])
    mean_date = df['DateOfBirth'].dropna().mean()
    df['DateOfBirth'].fillna(mean_date, inplace=True)
    columns = ['Gender', 'Symptoms', 'Causes', 'Disease', 'Medicine']
    for column in columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    df['Name'].fillna('Unknown Patient', inplace=True)

    gender_encoder = LabelEncoder()
    symptoms_encoder = LabelEncoder()
    causes_encoder = LabelEncoder()
    medicine_encoder = LabelEncoder()
    disease_encoder = LabelEncoder()

    # encoder.fit(df[['Gender', 'Symptoms', 'Causes', 'Medicine']])
    # encoded_array = encoder.transform([gender, symptoms, causes, medicine])

    gender_column_list = df['Gender'].tolist()
    gender_column_list.append(gender)
    gender_encoder.fit(gender_column_list)
    encoded_gender = gender_encoder.transform([gender])

    symptoms_column_list = df['Symptoms'].tolist()
    symptoms_column_list.append(symptoms)
    symptoms_encoder.fit(symptoms_column_list)
    encoded_symptoms = symptoms_encoder.transform([symptoms])

    causes_column_list = df['Causes'].tolist()
    causes_column_list.append(causes)
    causes_encoder.fit(causes_column_list)
    encoded_causes = causes_encoder.transform([causes])

    medicine_column_list = df['Medicine'].tolist()
    medicine_column_list.append(medicine)
    medicine_encoder.fit(medicine_column_list)
    encoded_medicine = medicine_encoder.transform([medicine])

    int_features = [[int(encoded_gender[0]), int(encoded_symptoms[0]), int(encoded_causes[0]), int(encoded_medicine[0])]]
    # features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model

    # model1 = joblib.load(open('rf_model_dis.joblib','rb'))
    print(int_features)
    prediction = model1.predict(int_features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    disease_column_list = df['Disease'].tolist()
    disease_encoder.fit(disease_column_list)
    original_labels = disease_encoder.inverse_transform([output])

    #####

    #Predicting 2

    # X = df['Symptoms'] + df['Causes'] + df['Medicine'] + df['Gender']
    # y = df.Disease
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # vectorizer = CountVectorizer(stop_words="english").fit(X_train)
    # df_train = pd.DataFrame(vectorizer.transform(X_train).todense(),
    #                         columns=vectorizer.get_feature_names_out())

    #####

    # Database

    conn = psycopg2.connect(
        host=config_details["database_host"],
        database="Hackomania2024",
        user="postgres",
        password=config_details["password"]
    )
    cur = conn.cursor()
    instruction = """
INSERT INTO medical_records (name, symptoms, disease, location) VALUES (%s, %s, %s, %s);
"""
    data = (name, symptoms, f"{original_labels[0]}", f"{coordinates}")

    try:
        cur.execute(instruction, data)
        conn.commit()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()


    return render_template('index1.html', prediction_text=f'Possible disease: {original_labels[0]}')

@app.route('/index2', methods=['GET', 'POST'])
def index2():
    return render_template("index2.html")

@app.route('/predict2', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, img_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None

@app.route('/symptoms_pred', methods=['GET', 'POST'])
def symptoms_pred():
    return render_template("symptoms_pred.html")

@app.route('/symptoms_result', methods=['GET', 'POST'])
def symptoms_result():
    if request.method == 'POST':
        result = request.form["Data"]
        result_pred = clf.predict(loaded_vec.transform([result]))
        decoded_result_pred = loaded_vec.inverse_transform([[0,2,3,1,23,3,4,4,2,1]])
        return render_template("symptoms_result.html", result = decoded_result_pred)

if __name__ == '__main__':
    app.run(debug=True)
