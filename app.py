from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle XGBoost et le scaler sauvegardés
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Charger les label encoders pour les colonnes catégorielles
with open('label_encoder_proto.pkl', 'rb') as file:
    label_encoder_proto = pickle.load(file)

with open('label_encoder_service.pkl', 'rb') as file:
    label_encoder_service = pickle.load(file)

with open('label_encoder_state.pkl', 'rb') as file:
    label_encoder_state = pickle.load(file)

with open('label_encoder_attack_cat.pkl', 'rb') as file:
    label_encoder_attack_cat = pickle.load(file)

# Route pour afficher la page d'accueil (index.html ou home.html)
@app.route('/')
def home():
    return render_template('home.html')

# Route pour effectuer une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées par l'utilisateur via une requête POST
        data = request.get_json()

        # Vérifiez si 'data' est une liste
        if 'data' in data and isinstance(data['data'], list):
            predictions = []
            for item in data['data']:
                # Créer un DataFrame à partir de l'item, mais exclure les colonnes non nécessaires
                input_data = pd.DataFrame([item]).drop(columns=['srcip', 'dstip', 'Djit', 'Stime', 'Ltime', 'Label'])

                # Appliquer l'encodage des colonnes catégorielles avec les encoders chargés
                input_data['proto'] = label_encoder_proto.transform(input_data['proto'])
                input_data['service'] = label_encoder_service.transform(input_data['service'])
                input_data['state'] = label_encoder_state.transform(input_data['state'])
                input_data['attack_cat'] = label_encoder_attack_cat.transform(input_data['attack_cat'])  # Encoder attack_cat

                # Vérifiez le nombre de colonnes
                print(f"Nombre de colonnes dans input_data : {input_data.shape[1]}")  # Affiche le nombre de colonnes

                # Appliquer la normalisation sur les données d'entrée
                input_scaled = scaler.transform(input_data)

                # Prédire la classe (type d'attaque)
                prediction = model.predict(input_scaled)
                predictions.append(int(prediction[0]))

            # Retourner les prédictions sous forme de réponse JSON
            return jsonify({'predictions': predictions})

        else:
            return jsonify({'error': 'Expected a list of data under the key "data"'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)