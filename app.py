from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('disease_predictor_rf.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Replace your SYMPTOMS list with the complete list from your training data
# This should match the columns in your original Training.csv (except the prognosis column)
SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
    'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 
    'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 
    'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 
    'altered_sensorium', 'red_spots_over_body', 'belly_pain', 
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 
    'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 
    'yellow_crust_ooze'
]


@app.route('/')
def home():
    return render_template('index.html', symptoms=SYMPTOMS)

# In app.py, replace the prediction route with this:

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get selected symptoms from form
        selected_symptoms = request.form.getlist('symptoms')
        
        # Create a complete feature vector with all zeros
        symptom_vector = [0] * len(SYMPTOMS)
        
        # Set to 1 for selected symptoms
        for symptom in selected_symptoms:
            if symptom in SYMPTOMS:
                index = SYMPTOMS.index(symptom)
                symptom_vector[index] = 1
        
        # Convert to numpy array and reshape
        symptoms_array = np.array(symptom_vector).reshape(1, -1)
        
        # Scale the features
        symptoms_scaled = scaler.transform(symptoms_array)
        
        # Make prediction
        prediction = model.predict(symptoms_scaled)
        
        # Get disease name
        disease = label_encoder.inverse_transform(prediction)[0]
        
        # Get probabilities
        probabilities = model.predict_proba(symptoms_scaled)[0]
        top3_indices = probabilities.argsort()[-3:][::-1]
        top3_diseases = label_encoder.inverse_transform(top3_indices)
        top3_probs = probabilities[top3_indices]
        
        return render_template('index.html', 
                             prediction=disease,
                             top_predictions=zip(top3_diseases, top3_probs),
                             symptoms=SYMPTOMS,
                             selected_symptoms=selected_symptoms)
    
    except Exception as e:
        return render_template('index.html', 
                             error=str(e),
                             symptoms=SYMPTOMS)

if __name__ == '__main__':
    app.run(debug=True)