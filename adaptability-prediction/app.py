from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__, template_folder="C:/Users/mzalendo_ke/Documents/Machine Learning/adaptability-prediction/templates")

# Load the saved stacking model
with open('stacking_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input data from the form
        input_data = {
            "Institution Type": request.form.get("institution_type"),
            "Load-shedding": request.form.get("load_shedding"),
            "Gender": request.form.get("gender"),
            "IT Student": request.form.get("it_student"),
            "Age": request.form.get("age"),
            "Network Type": request.form.get("network_type"),
            "Self Lms": request.form.get("self_lms"),
            "Financial Condition": request.form.get("financial_condition"),
            "Class Duration": request.form.get("class_duration"),
            "Education Level": request.form.get("education_level"),
            "Device": request.form.get("device"),
            "Internet Type": request.form.get("internet_type"),
            "Location": request.form.get("location"),
        }

        # Debugging: Print the input_data dictionary
        print("Form Input Data:", input_data)

        # Check if any field is missing (None)
        if None in input_data.values():
            return "Error: One or more form fields are empty."

        # Convert input data to DataFrame
        features = pd.DataFrame({
            key: [float(value)] for key, value in input_data.items()
        })

        # Debugging: Print the DataFrame
        print("Input DataFrame:\n", features)

        # Make prediction
        prediction = model.predict(features)[0]

        # Map prediction to adaptability level
        adaptability_map = {0: "Low", 1: "Medium", 2: "High"}
        adaptability = adaptability_map[prediction]

        return render_template("result.html", adaptability=adaptability)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
