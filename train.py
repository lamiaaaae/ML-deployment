import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.decomposition import PCA
import time
import joblib
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Charger les données
data = pd.read_csv("C:/Users/lenovo/Downloads/UNSW_NB15 (2).csv")
data.info()  # Afficher les informations sur les colonnes

# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne :\n", data.isnull().sum())

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Vérification des types de données
print("Types de données après nettoyage :\n", data.dtypes)

# Normalisation des colonnes numériques
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
print("Colonnes numériques :\n", num_cols)
scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Séparer les caractéristiques et la cible
X = data.drop("attack_cat", axis=1)  # Changer 'label' par 'attack_cat'
y = data["attack_cat"]  # Utiliser 'attack_cat' comme cible

# Affichage des premières lignes de X et y pour vérifier
print("Premières lignes de X :\n", X.head())
print("Premières lignes de y :\n", y.head())

# Encodage des colonnes non numériques dans X
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Vérification de X après encodage
print("Premières lignes de X après encodage :\n", X.head())

# Réduction de la taille des données (échantillonnage) (pour accélérer l'exécution)
X, y = X.sample(frac=0.01, random_state=42), y.sample(frac=0.01, random_state=42)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Taille des données d'entraînement :", X_train.shape, y_train.shape)
print("Taille des données de test :", X_test.shape, y_test.shape)

# Normalisation/Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Vérification de X_train et X_test après normalisation
print("Premières lignes de X_train après normalisation :\n", X_train[:5])
print("Premières lignes de X_test après normalisation :\n", X_test[:5])

# Réduction de la dimensionnalité avec PCA pour garder 95% de la variance
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Vérification de la forme de X après PCA
print(f"Dimensions après PCA : X_train_reduced {X_train_reduced.shape}, X_test_reduced {X_test_reduced.shape}")

# Sélectionner le modèle XGBoost
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), eval_metric='mlogloss', use_label_encoder=False)

# Entraîner et évaluer le modèle
try:
    print(f"Entraînement du modèle XGBoost...")
    start_time = time.time()  # Chronomètre pour l'entraînement
    model.fit(X_train_reduced, y_train)  # Entraînement du modèle
    end_time = time.time()  # Temps de fin
    print(f"Temps d'entraînement du modèle : {end_time - start_time:.2f} secondes")
    
    y_pred = model.predict(X_test_reduced)  # Prédiction sur l'ensemble de test

    # Afficher les résultats
    print(f"XGBoost Accuracy:", accuracy_score(y_test, y_pred))
    print(f"XGBoost Classification Report:\n", classification_report(y_test, y_pred))

    # Afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Matrice de confusion pour XGBoost")
    plt.show()

except Exception as e:
    print(f"Erreur lors de l'entraînement du modèle XGBoost : {e}")

# Enregistrement du modèle
try:
    joblib.dump(model, 'model_multiclass_xgboost.pkl')

    if os.path.exists('model_multiclass_xgboost.pkl'):
        print("Le fichier model_multiclass_xgboost.pkl a été créé avec succès.")
    else:
        print("Le fichier model_multiclass_xgboost.pkl n'a pas été créé.")

except Exception as e:
    print(f"Une erreur est survenue lors de l'enregistrement du modèle : {e}")

# Calculer la courbe ROC pour chaque classe
try:
    y_bin = label_binarize(y_test, classes=model.classes_)
    y_prob = model.predict_proba(X_test_reduced)

    for i in range(y_prob.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model.classes_[i]} (AUC = {roc_auc:.2f})')

except Exception as e:
    print(f"Erreur lors du calcul de la courbe ROC pour XGBoost : {e}")

# Tracer la diagonale
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Ajouter les labels et titre
plt.title('Courbe ROC pour XGBoost (Multiclasse)')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.legend(loc='lower right')
plt.show()
