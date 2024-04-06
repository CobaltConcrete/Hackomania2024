from flask import Flask, render_template, request, redirect, url_for
import pprint
import requests

app = Flask(__name__)
app.config['EXPLAIN_TEMPLATE_LOADING'] = True

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
            return render_template('form_page.html')
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
        results_geometry_bounds = gmaps_data_dict["results"][0]["geometry"]["bounds"]
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

if __name__ == '__main__':
    app.run(debug=True)
