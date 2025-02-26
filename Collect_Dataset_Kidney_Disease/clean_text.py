import pandas as pd
import re

# Charger les articles collectés
df = pd.read_csv("articles_maladies_renales.csv")

# Vérifier les premières lignes
print(df.head())
print("🔍 Nombre total d'articles :", len(df))


def clean_text(text):
    """Nettoie le texte en supprimant les caractères spéciaux et les espaces inutiles."""
    if pd.isna(text): return ""
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    text = re.sub(r'[^\w\s,.?!]', '', text)  # Garder uniquement lettres, chiffres et ponctuation
    text = text.strip().lower()
    return text

# Appliquer le nettoyage sur les abstracts en anglais et français
df["abstract_clean"] = df["abstract"].apply(clean_text)
df["abstract_fr_clean"] = df["abstract_fr"].apply(clean_text)

# Supprimer les lignes avec abstracts vides
df = df[df["abstract_clean"] != ""]
df = df[df["abstract_fr_clean"] != ""]

print("✅ len Données nettoyées !", len(df))