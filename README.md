# Disease Prediction System

An AI-powered healthcare application that predicts potential diseases based on user-selected symptoms using an Machine Learning model. The system combines Random Forest, XGBoost, and LightGBM through soft voting to improve prediction accuracy and provide reliable health insights.

> **Disclaimer:** This application is intended for educational and informational purposes only. It provides predictive insights and should not be considered a substitute for professional medical diagnosis or treatment.

## Features

* Smart symptom search and multi-select interface
* Ensemble Machine Learning model for improved prediction performance
* Top 3 disease predictions with confidence scores
* Symptom-based disease classification
* Suggested next steps based on prediction results
* RESTful API integration with Flask backend
* Modern and responsive user interface
* Fast and scalable prediction pipeline

## How It Works

### 1. Symptom Selection

Users search and select symptoms through an intuitive web interface.

### 2. Data Processing

Selected symptoms are converted into a binary feature vector compatible with the trained machine learning model.

### 3. Disease Prediction

An Ensemble Voting Classifier combines predictions from:

* Random Forest
* XGBoost
* LightGBM

### 4. Results Generation

The system returns:

* Top 3 predicted diseases
* Prediction probabilities
* Matched symptoms
* Suggested actions and recommendations

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask
* Flask-CORS
* REST APIs

### Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM
* Pandas
* NumPy

## Key Highlights

* Ensemble Learning for higher predictive accuracy
* Multi-class disease classification
* Real-time prediction generation
* Scalable API-driven architecture
* User-friendly healthcare assistance platform

## Future Enhancements

* Cloud deployment (AWS, Azure, Render)
* Mobile application support
* Deep Learning-based disease prediction
* Doctor recommendation system
* Personalized health analytics dashboard

## Author

**Abhishek Agarwal**
