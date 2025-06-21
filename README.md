**Emotion Detection & Interaction System for Autism Spectrum Disorder**

A Flask + frontend app that:

Detects faces and recognizes emotions in real time (up to 500 emotion classes).

Stores emotions temporarily in memory during the session.

Displays detected emotion history.

Includes a chatbot that interacts based on the live-emotion context.

1. Prerequisites
Python 3.6+ and pip

Webcam access

Internet connection for installing dependencies

2. Download the Project
git clone https://github.com/Mallika5c5/Emotion-Detection-and-Interaction-System-for-ASD-using-ML

3. Install Dependencies
   pip install -r requirements.txt
   This installs:

Flask (web server)

OpenCV (face detection)

FER or other emotion-recognition libraries

TensorFlow / Keras (backend model)

Any other listed dependencies

4. Launch the Application
    Start the Flask App
    python app.py
   
   You should see something like this appear:
    * Serving Flask app "app.py"
    * Debug mode: on
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

5. Open the Frontend
   http://127.0.0.1:5000/

   in your web browser. You’ll be greeted with:

      A Live webcam feed showing your face via OpenCV

      A Chatbot widget 

      A “Emotion History” button to check past emotions

 6. Using the App

        Allow your browser to access the camera.

        Ensure your face is visible in the live feed.

        The app will detect and display your emotion in real time.

        Each emotion is stored in memory for the session (no database).

        Click the History link to see all captured emotions along with timestamps.

        Use the Chatbot to interact, e.g.:

        ->How to make friends?


       
