import pandas as pd
import re
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi

# Charger les données
print("📊 Chargement des données...")
df = pd.read_csv("imdb_top_500.csv")

# Prétraitement du texte
def clean_text(text):
    if pd.isna(text):
        return ""
    # Enlever les balises HTML
    text = re.sub(r'<[^>]+>', ' ', str(text))
    # Enlever les caractères spéciaux et mettre en minuscule
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Enlever les espaces multiples
    text = ' '.join(text.split())
    return text

print("🔧 Prétraitement des données...")
df['clean_text'] = df['text'].apply(clean_text)

# Préparation des données
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorisation + Modèle
print("🤖 Entraînement du modèle...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Évaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy sur le test set : {accuracy:.4f}")

# Sauvegarde des modèles
print("💾 Sauvegarde des modèles...")
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

config = {
    "model_type": "LogisticRegression",
    "vectorizer": "TfidfVectorizer",
    "max_features": 5000,
    "ngram_range": [1, 2],
    "accuracy": round(accuracy, 4),
    "features": ["clean_text"],
    "target": "label"
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("📦 Modèles sauvegardés : model.pkl, vectorizer.pkl, config.json")

# Upload vers Hugging Face
hf_token = os.environ.get("HF_TOKEN")
hf_repo_id = os.environ.get("HF_REPO_ID")

if hf_token and hf_repo_id:
    print(f"🚀 Upload vers Hugging Face : {hf_repo_id}")
    
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        
        for file in ["model.pkl", "vectorizer.pkl", "config.json"]:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=hf_repo_id,
                repo_type="model"
            )
            print(f"✅ {file} uploadé avec succès !")
        
        print(f"🎉 Modèle disponible sur : https://huggingface.co/{hf_repo_id}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload HF : {e}")
else:
    print("ℹ️ HF_TOKEN ou HF_REPO_ID non détecté (normal en local)")