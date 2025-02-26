import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import pandas as pd
from clean_text import clean_text

# Charger les articles collectés
df = pd.read_csv("articles_maladies_renales.csv")



# Charger les données en Pandas et les convertir en dataset Hugging Face
def prepare_dataset(df):
    data = {
        "context": df["abstract_fr_clean"].tolist(),
        "question": df["qcm"].tolist()
    }
    return Dataset.from_dict(data)

# Charger et tokenizer les données
def tokenize_function(examples, tokenizer):
    inputs = [f"Contexte: {c}\nQuestion: {q}" for c, q in zip(examples["context"], examples["question"])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)


# Appliquer le nettoyage sur les abstracts en anglais et français
df["abstract_clean"] = df["abstract"].apply(clean_text)
df["abstract_fr_clean"] = df["abstract_fr"].apply(clean_text)

# Supprimer les lignes avec abstracts vides
df = df[df["abstract_clean"] != ""]
df = df[df["abstract_fr_clean"] != ""]

# Configuration du modèle et du tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")



# Charger le dataset (remplace 'df' par ton DataFrame contenant les données)
dataset = prepare_dataset(df)
tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

# Séparation train/test
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset, eval_dataset = train_test_split["train"], train_test_split["test"]



# Paramètres d'entraînement
training_args = TrainingArguments(
    output_dir="./llama2_qcm_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
    logging_dir="./logs",
)

# Préparer le data collator pour le padding
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

# Lancer l'entraînement
def train_model():
    trainer.train()
    model.save_pretrained("./llama2_qcm_finetuned")
    tokenizer.save_pretrained("./llama2_qcm_finetuned")
    print("✅ Fine-tuning terminé et modèle sauvegardé !")

# Génération de QCM après fine-tuning
def generate_qcm_with_llama2(context):
    """Génère une question QCM à partir du contexte médical avec LLaMA 2 fine-tuné"""
    prompt = f"Contexte: {context}\nGénère une question à choix multiples sur ce sujet."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exemple d'utilisation
if __name__ == "__main__":
    train_model()
    test_context = "L'insuffisance rénale chronique est une maladie progressive affectant les reins."
    print(generate_qcm_with_llama2(test_context))
