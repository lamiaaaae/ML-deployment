import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle  # Importer pickle pour la sauvegarde du modèle

# Charger le dataset
df = pd.read_csv('C:/Users/lenovo/Downloads/UNSW_NB15 (2).csv')  # Remplacez par le chemin de votre dataset

# Afficher les premières lignes du dataset
print(df.head())
print(df.info())

# Étape 1: Nettoyage du dataset
# 1.1 Vérifier les valeurs manquantes
print(df.isnull().sum())

# 1.2 Supprimer les lignes avec des valeurs manquantes (ou remplissez-les)
df = df.dropna()  # ou df.fillna(méthode='ffill') pour remplir les valeurs manquantes

# Étape 2: Encodage des variables catégorielles
label_encoder_proto = LabelEncoder()
label_encoder_service = LabelEncoder()
label_encoder_state = LabelEncoder()
label_encoder_attack_cat = LabelEncoder()  # Ajout de l'encodeur pour attack_cat

# Encoder les variables catégorielles
df['proto'] = label_encoder_proto.fit_transform(df['proto'])
df['service'] = label_encoder_service.fit_transform(df['service'])
df['state'] = label_encoder_state.fit_transform(df['state'])
df['attack_cat'] = label_encoder_attack_cat.fit_transform(df['attack_cat'])  # Encoder attack_cat

# Vérifier les classes uniques dans les label encoders
print("Classes uniques pour 'proto':", label_encoder_proto.classes_)
print("Classes uniques pour 'service':", label_encoder_service.classes_)
print("Classes uniques pour 'state':", label_encoder_state.classes_)
print("Classes uniques pour 'attack_cat':", label_encoder_attack_cat.classes_)  # Afficher les classes uniques pour attack_cat

# Sauvegarder les label encoders pour chaque colonne catégorielle
with open('label_encoder_proto.pkl', 'wb') as file:
    pickle.dump(label_encoder_proto, file)

with open('label_encoder_service.pkl', 'wb') as file:
    pickle.dump(label_encoder_service, file)

with open('label_encoder_state.pkl', 'wb') as file:
    pickle.dump(label_encoder_state, file)

with open('label_encoder_attack_cat.pkl', 'wb') as file:  # Sauvegarder l'encodeur pour attack_cat
    pickle.dump(label_encoder_attack_cat, file)

# Vérifier les transformations
print(df[['proto', 'service', 'state', 'attack_cat']].head())

# Étape 3: Séparer les données en variables indépendantes (X) et variable cible (y)
X = df.drop(['attack_cat', 'label'], axis=1)  # Exclure 'attack_cat' et 'label' de X
y = df['attack_cat']  # La variable cible est 'attack_cat'

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4: Normalisation des données
scaler = StandardScaler()

# Appliquer la standardisation sur les données d'entraînement et de test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Étape 5: Entraîner le modèle XGBoost
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(y.unique()), random_state=42)

# Entraîner le modèle
model.fit(X_train_scaled, y_train)

# Sauvegarder le modèle avec Pickle
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Sauvegarder également l'objet scaler si tu veux l'utiliser lors de la prédiction future
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Étape 6: Évaluation du modèle
# Prédire les classes sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Afficher le rapport de classification
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder_proto.classes_, yticklabels=label_encoder_proto.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Étape 7: Afficher l'importance des caractéristiques
plt.barh(X.columns, model.feature_importances_)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from XGBoost')
plt.show()