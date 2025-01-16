from flask import Flask, request, render_template  # Import necessary modules from Flask for routing and rendering HTML templates
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for handling data in DataFrame format

# Import the custom data and prediction pipeline modules
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling (though not used in this snippet)
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Import the CustomData class (for feature input) and PredictPipeline class (for model prediction)

# Create a Flask application instance
application = Flask(__name__)

# Assigning the Flask app to 'app' for shorthand use (commonly used convention)
app = application

## Route for the home page (landing page)
@app.route('/')
def index():
    # Render the index.html template when the home route ('/') is accessed
    return render_template('index.html')

# Route for prediction - handles both GET and POST requests
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If the request is a GET, return the home page where users can input their data
        return render_template('home.html')
    else:
        # If the request is a POST (when the form is submitted), extract form data
        data = CustomData(
            gender=request.form.get('gender'),  # Get the 'gender' field value from the form
            race_ethnicity=request.form.get('ethnicity'),  # Get 'ethnicity' from the form
            parental_level_of_education=request.form.get('parental_level_of_education'),  # Get 'parental_level_of_education'
            lunch=request.form.get('lunch'),  # Get 'lunch' (whether the person has a standard or free/reduced lunch)
            test_preparation_course=request.form.get('test_preparation_course'),  # Get 'test_preparation_course' info
            reading_score=float(request.form.get('writing_score')),  # Convert 'writing_score' input to float
            writing_score=float(request.form.get('reading_score'))  # Convert 'reading_score' input to float
        )

        # Convert the form data into a pandas DataFrame for prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Debug: print the DataFrame for inspection
        print("Before Prediction")

        # Instantiate the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # Use the pipeline to make predictions based on the input data
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render the 'home.html' template again, passing the prediction result to display on the page
        return render_template('home.html', results=results[0])  # 'results[0]' corresponds to the first predicted result

# The application runs here, making it accessible over the network
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug= True)  # Run the app with the host set to '0.0.0.0', which makes the app publicly accessible
