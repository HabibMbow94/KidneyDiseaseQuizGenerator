import pandas as pd
import re

# Charger les articles collectés
df = pd.read_csv("articles_maladies_renales.csv")

# Vérifier les premières lignes
print(df.head())
# print("🔍 Nombre total d'articles :", len(df))

def clean_text(text):
    """Nettoie le texte en supprimant les caractères spéciaux et les espaces inutiles."""
    if pd.isna(text): return ""
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    text = re.sub(r'[^\w\s,.?!]', '', text)  # Garder uniquement lettres, chiffres et ponctuation
    text = text.strip().lower()
    return text

# Appliquer le nettoyage sur les abstracts en français
df["abstract_fr_clean"] = df["abstract_fr"].apply(clean_text)

# Supprimer les lignes avec abstracts vides
df = df[df["abstract_fr_clean"] != ""]

# Créer la liste des abstracts propres
list_abstract_fr_clean = df["abstract_fr_clean"].tolist()

# print(f"✅ Données nettoyées ! Nombre d'abstracts : {len(list_abstract_fr_clean)}")