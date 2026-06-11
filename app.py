
"""
Flask backend API for Disease Prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import pandas as pd
import time

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

model = None
symptoms_list = None
diseases_list = None


# ---------------- LOAD ----------------
def load_model():
    global model, symptoms_list, diseases_list

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('symptoms_list.json') as f:
        symptoms_list = json.load(f)

    with open('diseases_list.json') as f:
        diseases_list = json.load(f)

    print("[SUCCESS] Model + Data Loaded")


# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route('/symptoms')
def get_symptoms():
    return jsonify({
        "status": "success",
        "symptoms": symptoms_list
    })


@app.route('/diseases')
def get_diseases():
    return jsonify({
        "status": "success",
        "diseases": diseases_list
    })


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        start = time.time()

        data = request.get_json()
        user_symptoms = data.get("symptoms", [])

        # Create DataFrame (IMPORTANT FIX)
        input_df = pd.DataFrame(
            [[0]*len(symptoms_list)],
            columns=symptoms_list
        )

        matched = []

        for s in user_symptoms:
            s = s.lower().strip()

            if s in symptoms_list:
                input_df.loc[0, s] = 1
                matched.append(s)

            else:
                for sym in symptoms_list:
                    if s in sym:
                        input_df.loc[0, sym] = 1
                        matched.append(sym)
                        break

        if not matched:
            return jsonify({"error": "No matching symptoms"}), 400

        probs = model.predict_proba(input_df)[0]
        top_idx = probs.argsort()[-3:][::-1]

        top_predictions = []
        for i in top_idx:
            top_predictions.append({
                "disease": diseases_list[i],
                "probability": round(float(probs[i])*100, 2)
            })

        best = top_idx[0]

        end = time.time()

        return jsonify({
            "status": "success",
            "prediction": {
                "disease": diseases_list[best],
                "probability": round(float(probs[best])*100, 2)
            },
            "top_predictions": top_predictions,
            "matched_symptoms": matched,
            "suggested_steps": "Consult a doctor for confirmation.",
            "response_time": round(end-start, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("[INFO] Starting server...")
    load_model()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)