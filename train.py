import pandas as pd
import re
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================
# C'EST ICI QUE TU METS LA LIGNE :
# ============================================
df = pd.read_csv("imdb_top_500.csv")  # chemin relatif, pas de dossier
# ============================================

# Ensuite, tu continues avec le prétraitement
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text = ' '.join(text.split())
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Préparation des données
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorisation + Modèle
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Évaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy : {accuracy:.4f}")

# Sauvegarde
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

config = {
    "model_type": "LogisticRegression",
    "vectorizer": "TfidfVectorizer",
    "max_features": 5000,
    "accuracy": round(accuracy, 4)
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("💾 Modèles sauvegardés")

import os
from huggingface_hub import HfApi

# Récupération des variables d'environnement (fournies par GitHub Actions)
hf_token = os.environ.get("HF_TOKEN")
hf_repo_id = os.environ.get("HF_REPO_ID")

if hf_token and hf_repo_id:
    print(f"🚀 Tentative d'upload vers Hugging Face : {hf_repo_id}")
    
    try:
        api = HfApi(token=hf_token)
        # Créer le repo si il n'existe pas encore
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        
        # Uploader les fichiers un par un
        for file in ["model.pkl", "vectorizer.pkl", "config.json"]:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=hf_repo_id,
                repo_type="model"
            )
            print(f"✅ {file} uploadé !")
            
        print(f"🎉 Modèle disponible sur : https://huggingface.co/{hf_repo_id}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload HF : {e}")
else:
    print("ℹ️ Pas de token ou ID repo détecté (normal si test en local).")