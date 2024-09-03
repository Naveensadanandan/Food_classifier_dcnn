import streamlit as st

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    from flask import Flask, request, send_file, jsonify
    import os
    from flask_cors import CORS
    from PIL import Image
    import torch
    from util import pred_plot
    from util import loaded_model
    from util import custom_image_transform

    # Initialize the Flask app
    app = Flask(__name__)
    CORS(app)

    # Path to the model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load your trained model
    loaded_model.eval()

    # Define image transformations (adjust based on your model's requirements)

    @app.route('/predict', methods=['POST'])
    def predict():
        """Handle the prediction request and return the plot image."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename.lower().endswith(('.jpg', '.jpeg')):
            # Save the uploaded file to a temporary location
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Generate the prediction plot
            title= pred_plot(
                model=loaded_model,
                image_path=file_path,
                class_names=['apple_pie', 'chicken_wings', 'pizza', 'steak', 'sushi'],
                transform=custom_image_transform,
                device=device
            )

            # Remove the uploaded file after processing
            os.remove(file_path)

            # Send the generated plot image as a response
            return title

        else:
            return jsonify({'error': 'Invalid file type. Only JPEG images are allowed.'}), 400

    if __name__ == '__main__':
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        app.run()

    app.run(port=8888)


# We'll never reach this part of the code the first time this file executes!

import streamlit as st
import requests
from PIL import Image
import io

# Streamlit web app title
st.title('Food Image Classifier')

st.markdown("""
This app is for educational purposes only and has been trained to recognize only five food items:
- Apple Pie
- Chicken Wings
- Pizza
- Sushi
- Steak

Please upload images related to these items for classification.
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose a JPEG image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Show a button to make the prediction
    if st.button('Classify'):
        # Display a message while processing
        st.write("Classifying...")

        # Convert the uploaded file to a bytes object
        files = {"file": uploaded_file.getvalue()}

        # Make a POST request to the Flask API
        try:
            response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded_file})

            if response.status_code == 200:
                # Display the prediction returned from the API
                prediction = response.json().get('prediction', 'No prediction returned')
                st.write(f"Prediction: {prediction}")
            else:
                # Display error message
                st.write(f"Error: {response.status_code}")
                st.write(response.json().get('error', 'Unknown error'))

        except requests.exceptions.RequestException as e:
            st.write(f"Error connecting to API: {e}")
