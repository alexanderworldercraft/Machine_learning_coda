# Importation des bibliothèques nécessaires
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import glob

# Définir les chemins des fichiers de données et des images
DATA_PATH_TRAIN = './asset/data/train_data.csv'
DATA_PATH_TEST = './asset/data/test_data.csv'
IMAGE_PATH_TRAIN = './asset/data/train_photos/'
IMAGE_PATH_TEST = './asset/data/test_photos/'

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

# Fonction pour charger et prétraiter les images avec PyTorch
def load_image_features(pet_id, image_dir, model, transform, image_size=(224, 224)):
    """Charge et prétraite les images associées à un PetID, et extrait les caractéristiques visuelles à partir d'un modèle pré-entraîné."""
    image_features = []
    image_paths = glob.glob(os.path.join(image_dir, f"{pet_id}-1.jpg"))  # On enlève le préfixe `._`
    
    if not image_paths:
        print(f"Aucune image trouvée pour {pet_id}")
        return np.zeros(2048)  # Retourne un vecteur nul si aucune image n'est trouvée

    for img_path in image_paths:
        if os.path.exists(img_path):
            print(f"Chargement de l'image : {img_path}")
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                img = img.unsqueeze(0)  # Ajouter une dimension batch

                # Mettre le modèle en mode évaluation
                model.eval()
                with torch.no_grad():
                    features = model(img).flatten().numpy()  # Extraire les features

                image_features.append(features)
            except Exception as e:
                print(f"Erreur lors du chargement de l'image {img_path}: {e}")
    
    if len(image_features) > 0:
        return np.mean(image_features, axis=0)
    else:
        return np.zeros(2048)

# Chargement du modèle ResNet50 pré-entraîné de PyTorch
resnet_model = models.resnet50(pretrained=True)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # Retirer la dernière couche (classification)

# Transformation des images pour correspondre à la pré-formation du modèle ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extraire les caractéristiques des images associées aux animaux dans le train_data
print("Extraction des caractéristiques des images pour l'entraînement...")
train_image_features = []
for pet_id in train_df['PetID']:
    features = load_image_features(pet_id, IMAGE_PATH_TRAIN, resnet_model, transform)
    train_image_features.append(features)

# Conversion en DataFrame
train_image_features_df = pd.DataFrame(train_image_features, columns=[f"img_feature_{i}" for i in range(2048)])

# Ajouter les caractéristiques visuelles aux données d'entraînement
train_df = pd.concat([train_df, train_image_features_df], axis=1)

# Sélection des features numériques, catégoriques et visuelles
numerical_features = ["Age", "FurLength", "Quantity", "Fee", "PhotoAmt", "VideoAmt"]
categorical_features = ["Vaccinated", "Dewormed", "Sterilized", "Health", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize"]
image_features = [f"img_feature_{i}" for i in range(2048)]
target = "AdoptionSpeed"

# Division des données en ensemble d'entraînement et de validation
print("Séparation des données en ensemble d'entraînement et de validation...")
X = train_df[numerical_features + categorical_features + list(tfidf_df.columns) + image_features]
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle avec validation croisée et recherche d'hyperparamètres pour RandomForestClassifier
print("Recherche des meilleurs hyperparamètres avec GridSearchCV pour RandomForestClassifier...")
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(5), 
                           scoring='accuracy', n_jobs=-1, verbose=1)
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
description_tfidf_test = tfidf.transform(test_df['Description'].astype(str))
tfidf_test_df = pd.DataFrame(description_tfidf_test.toarray(), columns=[f"tfidf_{i}" for i in range(description_tfidf_test.shape[1])])

# Ajout des nouvelles features basées sur TF-IDF aux données de test
test_df = pd.concat([test_df, tfidf_test_df], axis=1)

# Extraction des caractéristiques des images associées aux animaux dans le test_data
print("Extraction des caractéristiques des images pour les données de test...")
test_image_features = []
for pet_id in test_df['PetID']:
    features = load_image_features(pet_id, IMAGE_PATH_TEST, resnet_model, transform)
    test_image_features.append(features)

# Conversion en DataFrame
test_image_features_df = pd.DataFrame(test_image_features, columns=[f"img_feature_{i}" for i in range(2048)])

# Ajouter les caractéristiques visuelles aux données de test
test_df = pd.concat([test_df, test_image_features_df], axis=1)

# Prédictions sur l'ensemble de test
X_test = test_df[numerical_features + categorical_features + list(tfidf_test_df.columns) + image_features]
print("Prédictions sur les données de test...")
y_test_preds = best_model.predict(X_test)

# Sauvegarde des résultats
test_preds = test_df.copy()
test_preds[target] = y_test_preds
submission_df = test_preds[["PetID", target]]  # PetID étant l'identifiant dans le fichier test

# Sauvegarde dans un fichier CSV
submission_path = './asset/data/submission_image_RandomForestClassifier_pytorch.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Prédictions sauvegardées dans {submission_path}")

print("Fin de l'exécution.")