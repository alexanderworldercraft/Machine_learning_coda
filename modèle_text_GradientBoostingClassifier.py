# Importation des bibliothèques nécessaires
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Définir le chemin du fichier de données
DATA_PATH_TRAIN = './asset/data/train_data.csv'
DATA_PATH_TEST = './asset/data/test_data.csv'

# Chargement des données d'entraînement
print("Chargement des données d'entraînement...")
train_df = pd.read_csv(DATA_PATH_TRAIN)

# Vérification du chargement
print(f"Nombre de lignes dans le dataset d'entraînement : {len(train_df)}")
print(train_df.head())

# Traitement des descriptions (utilisation de TF-IDF sur les données d'entraînement)
print("Traitement des descriptions avec TF-IDF...")
tfidf = TfidfVectorizer(max_features=500)  # Limitation à 500 features pour éviter trop de dimensions
description_tfidf = tfidf.fit_transform(train_df['Description'].astype(str))

# Ajout des nouvelles features basées sur TF-IDF
tfidf_df = pd.DataFrame(description_tfidf.toarray(), columns=[f"tfidf_{i}" for i in range(description_tfidf.shape[1])])
train_df = pd.concat([train_df, tfidf_df], axis=1)

# Sélection des features numériques et catégoriques
numerical_features = ["Age", "FurLength", "Quantity", "Fee", "PhotoAmt", "VideoAmt"]
categorical_features = ["Vaccinated", "Dewormed", "Sterilized", "Health", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize"]
target = "AdoptionSpeed"

# Division des données en ensemble d'entraînement et de validation
print("Séparation des données en ensemble d'entraînement et de validation...")
X = train_df[numerical_features + categorical_features + list(tfidf_df.columns)]
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle avec validation croisée et recherche d'hyperparamètres
print("Recherche des meilleurs hyperparamètres avec GridSearchCV...")
model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [2, 4, 8]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(5), 
                           scoring='accuracy', n_jobs=-1, verbose=1)  # Utilisation de n_jobs=-1 pour utiliser tous les cœurs du CPU
grid_search.fit(X_train, y_train)

# Meilleurs paramètres et performance sur l'ensemble de validation
print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

print(f"Score de validation : {accuracy_score(y_val, y_val_pred)}")

# Chargement des données de test
print("Chargement des données de test...")
test_df = pd.read_csv(DATA_PATH_TEST)

# Application du même TF-IDF sur les données de test
print("Application de TF-IDF sur les données de test...")
description_tfidf_test = tfidf.transform(test_df['Description'].astype(str))  # Utilisation de transform (et non fit_transform)
tfidf_test_df = pd.DataFrame(description_tfidf_test.toarray(), columns=[f"tfidf_{i}" for i in range(description_tfidf_test.shape[1])])

# Ajout des nouvelles features basées sur TF-IDF aux données de test
test_df = pd.concat([test_df, tfidf_test_df], axis=1)

# Prédictions sur l'ensemble de test
X_test = test_df[numerical_features + categorical_features + list(tfidf_test_df.columns)]
print("Prédictions sur les données de test...")
y_test_preds = best_model.predict(X_test)

# Sauvegarde des résultats
test_preds = test_df.copy()
test_preds[target] = y_test_preds
submission_df = test_preds[["PetID", target]]  # PetID étant l'identifiant dans le fichier test

# Sauvegarde dans un fichier CSV
submission_path = './asset/data/submission_text_GradientBoostingClassifier.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Prédictions sauvegardées dans {submission_path}")

print("Fin de l'exécution.")