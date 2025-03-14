# import streamlit as st
# import requests
# import json
# import time
# from langchain.prompts import PromptTemplate

# # Configuration de l'API Ollama
# OLLAMA_URL = "http://localhost:11434/api/generate"

# def query_ollama(prompt, model="mistral", max_retries=3, wait_time=5):
#     """
#     Envoie une requête à l'API Ollama et gère les erreurs.
#     """
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         # Paramètres optionnels pour contrôler la génération
#         "options": {
#             "temperature": 0.1,
#             "top_p": 0.95,
#             "max_tokens": 2000,
#             "stop": ["\n\n\n"]  # Aide à éviter des tokens supplémentaires après la fin du JSON
#         },
#         "stream": False  # S'assurer que la réponse est reçue en une seule fois
#     }
    
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            
#             if response.status_code == 503:
#                 st.warning(f"Le modèle est en cours de chargement, attente de {wait_time} secondes...")
#                 time.sleep(wait_time)
#                 continue
                
#             response.raise_for_status()
            
#             # Récupérer le texte de la réponse
#             response_json = response.json()
#             if 'response' in response_json:
#                 return response_json['response']
#             else:
#                 st.error(f"Format de réponse inattendu: {response_json}")
#                 return None
            
#         except requests.exceptions.Timeout:
#             st.warning(f"Timeout lors de la tentative {attempt+1}/{max_retries}. Réessai...")
#             time.sleep(wait_time)
#         except Exception as e:
#             st.error(f"Tentative {attempt+1}/{max_retries} échouée: {str(e)}")
#             if attempt == max_retries - 1:
#                 return None
#             time.sleep(wait_time)
# def create_prompt_with_langchain(topic, number, difficulty):
#     """
#     Génère un prompt optimisé pour Ollama en utilisant LangChain.
#     """
#     template = PromptTemplate.from_template("""
#     Vous êtes un expert médical en néphrologie. 
#     Générez exactement {number} questions de niveau {difficulty} sur le sujet suivant : "{topic}".

#     IMPORTANT: Formatez UNIQUEMENT votre réponse en JSON valide sous cette structure exacte, sans aucun texte avant ou après le JSON:
    
#     {{
#         "1": {{
#             "question": "Question ici",
#             "options": {{
#                 "a": "Option A",
#                 "b": "Option B",
#                 "c": "Option C",
#                 "d": "Option D"
#             }},
#             "correct": "a",
#             "explanation": "Explication détaillée de la bonne réponse."
#         }},
#         "2": {{ ... }}
#     }}
    
#     Ne dites rien d'autre que le JSON. Assurez-vous que votre réponse est un JSON parfaitement valide sans commentaires supplémentaires.
#     """)
    
#     return template.format(topic=topic, number=number, difficulty=difficulty)

# def extract_json_from_text(text):
#     """
#     Extrait proprement le JSON de la réponse du modèle.
#     """
#     if not text:
#         return {"error": "Aucune réponse obtenue du modèle"}
        
#     # Trouver toutes les accolades ouvrantes et fermantes pour trouver le JSON complet
#     json_start = text.find('{')
#     json_end = text.rfind('}') + 1
    
#     if json_start >= 0 and json_end > json_start:
#         json_text = text[json_start:json_end]
#         try:
#             # Tentative de chargement direct
#             return json.loads(json_text)
#         except json.JSONDecodeError as e:
#             st.error(f"Erreur de parsing JSON: {e}")
            
#             # Tentative de nettoyage supplémentaire
#             try:
#                 # Parfois, il y a des caractères d'échappement ou des nouvelles lignes qui causent des problèmes
#                 import re
#                 # Supprimer les caractères d'échappement incorrects qui pourraient briser le JSON
#                 cleaned_json = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', json_text)
#                 return json.loads(cleaned_json)
#             except Exception:
#                 # Si tout échoue, afficher le texte pour le débogage
#                 st.code(json_text, language="json")
#                 return {"error": "Le JSON retourné par le modèle est invalide"}
#     else:
#         # Afficher le texte brut pour débogage
#         st.text("Réponse brute:")
#         st.code(text)
#         return {"error": "Aucun JSON trouvé dans la réponse"}

# debug_mode = st.sidebar.checkbox("Mode debug")
# # Dans votre fonction generate_mcq, ajoutez ceci:
# def generate_mcq(topic, number=5, difficulty="Moyen", model="mistral"):
#     """
#     Génère un QCM médical avec Ollama et retourne le JSON parsé.
#     """
#     prompt = create_prompt_with_langchain(topic, number, difficulty)
#     response_text = query_ollama(prompt, model)
#     if not response_text:
#         return {"error": "Pas de réponse du modèle après plusieurs tentatives"}
    
#     st.session_state["raw_response"] = response_text
    
#     # Afficher la réponse brute en mode debug
#     if debug_mode:
#         st.subheader("Réponse brute du modèle")
#         st.text(response_text)
    
#     return extract_json_from_text(response_text)

# def display_interactive_quiz(quiz_data):
#     """
#     Affiche un quiz interactif avec des boutons radio pour chaque question
#     et permet de soumettre les réponses pour évaluation.
#     """
#     if not quiz_data or "error" in quiz_data:
#         st.error(f"Erreur: {quiz_data.get('error', 'Données de quiz invalides')}")
#         return
    
#     # Initialiser les réponses de l'utilisateur si ce n'est pas déjà fait
#     if "user_answers" not in st.session_state:
#         st.session_state["user_answers"] = {}
    
#     # Initialiser l'état de soumission
#     if "quiz_submitted" not in st.session_state:
#         st.session_state["quiz_submitted"] = False
    
#     # Afficher chaque question avec des boutons radio
#     for q_num, q_data in quiz_data.items():
#         # st.markdown(f"### Question {q_num}")
        
#         # Afficher la question
#         question_text = q_data.get("question", q_data.get("mcq", "Question manquante"))
#         st.markdown(f"**{question_text}**")
        
#         # Afficher les options comme boutons radio
#         options = q_data.get("options", {})
#         option_keys = list(options.keys())
#         option_values = [f"{key}: {options[key]}" for key in option_keys]
        
#         # Créer un identifiant unique pour chaque question
#         q_id = f"q_{q_num}"
        
#         # Afficher les boutons radio
#         user_choice = st.radio(
#             "Sélectionnez votre réponse:",
#             option_values,
#             key=q_id,
#             index=None,
#             disabled=st.session_state["quiz_submitted"]
#         )
        
#         # Enregistrer la réponse de l'utilisateur
#         if user_choice:
#             selected_key = user_choice.split(":")[0].strip()
#             st.session_state["user_answers"][q_num] = selected_key
            
#         st.divider()
    
#     # Bouton de soumission
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if not st.session_state["quiz_submitted"]:
#             if st.button("Soumettre vos réponses", 
#                          use_container_width=True, 
#                          disabled=len(st.session_state["user_answers"]) < len(quiz_data)):
#                 st.session_state["quiz_submitted"] = True
#                 st.rerun()
#         else:
#             if st.button("Réinitialiser le quiz", use_container_width=True):
#                 st.session_state["quiz_submitted"] = False
#                 st.session_state["user_answers"] = {}
#                 st.rerun()

# def display_quiz_results(quiz_data):
#     """
#     Affiche les résultats du quiz avec les réponses correctes et les explications.
#     """
#     if not st.session_state["quiz_submitted"]:
#         return
    
#     st.markdown("## 📊 Résultats du Quiz")
    
#     correct_count = 0
#     total_questions = len(quiz_data)
    
#     for q_num, q_data in quiz_data.items():
#         user_answer = st.session_state["user_answers"].get(q_num, "")
#         correct_answer = q_data.get("correct", "")
        
#         # st.markdown(f"### Question {q_num}")
#         question_text = q_data.get("question", q_data.get("mcq", ""))
#         st.markdown(f"**{question_text}**")
        
#         # Vérifier si la réponse est correcte
#         is_correct = user_answer == correct_answer
#         if is_correct:
#             correct_count += 1
        
#         # Afficher le résultat de cette question
#         if is_correct:
#             st.success(f"✅ Votre réponse ({user_answer}) est correcte!")
#         else:
#             st.error(f"❌ Votre réponse ({user_answer}) est incorrecte. La réponse correcte est: {correct_answer}")
        
#         # Afficher l'explication
#         explanation_key = "explanation" if "explanation" in q_data else "explication"
#         if explanation_key in q_data:
#             st.info(f"**Explication**: {q_data[explanation_key]}")
        
#         # Afficher toutes les options
#         st.markdown("**Options:**")
#         options = q_data.get("options", {})
#         for opt_key, opt_value in options.items():
#             prefix = "✅ " if opt_key == correct_answer else "   "
#             highlight = "**" if opt_key == user_answer else ""
#             st.markdown(f"{prefix} {highlight}{opt_key}: {opt_value}{highlight}")
        
#         st.divider()
    
#     # Afficher le score global
#     score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
#     # Utiliser des colonnes pour un affichage plus élégant du score
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.markdown(f"## Score final: {correct_count}/{total_questions} ({score_percentage:.1f}%)")
        
#         # Évaluer la performance
#         if score_percentage >= 80:
#             st.balloons()
#             st.success("🎉 Excellent! Vous maîtrisez bien ce sujet!")
#         elif score_percentage >= 60:
#             st.success("👍 Bon travail! Vous avez une bonne compréhension du sujet.")
#         elif score_percentage >= 40:
#             st.warning("🔍 Pas mal, mais il y a encore des points à améliorer.")
#         else:
#             st.error("📚 Ce sujet nécessite plus de révision. Ne vous découragez pas!")

# # Style CSS personnalisé
# st.markdown("""
#     <style>
#         /* 💎 Background et design */
#         body {
#             background-color: #FBF8F1;
#             font-family: 'Arial', sans-serif;
#         }
#         .main-header {
#             text-align: center;
#             color: #EF4D89;
#             font-size: 2.5rem;
#             font-weight: bold;
#             padding: 1rem 0;
#             margin-bottom: 1rem;
#         }
#         .subtitle {
#             text-align: center;
#             color: #00BAC6;
#             font-size: 1.5rem;
#             margin-bottom: 2rem;
#             border-top: 2px solid #00BAC6;
#         }
#         .stButton>button {
#             background-color: #EF4D89 !important;
#             color: white !important;
#             font-size: 1.1rem !important;
#             font-weight: bold !important;
#             border-radius: 10px !important;
#             padding: 10px !important;
#         }
#         .stButton>button:hover {
#             background-color: #D43F76 !important;
#         }
#         .stRadio > label {
#             font-size: 1rem;
#             font-weight: bold;
#             color: #34495E;
#         }
#         .footer {
#             text-align: center;
#             margin-top: 2rem;
#             padding-top: 1rem;
#             # border-top: 2px solid #00BAC6;
#             font-size: 0.9rem;
#             color: #95A5A6;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # 🚀 **Interface Streamlit**
# st.markdown("<h1 class='main-header'>AKA CARE</h1>", unsafe_allow_html=True)
# st.markdown("<h2 class='subtitle'>Quiz Éducatif sur les Maladies Rénales Chroniques</h2>", unsafe_allow_html=True)


# # Initialiser l'état de session si nécessaire
# if "raw_response" not in st.session_state:
#     st.session_state["raw_response"] = ""
# if "quiz_data" not in st.session_state:
#     st.session_state["quiz_data"] = None

# # Section de l'en-tête et configuration
# with st.sidebar:
#     st.markdown("## Générateur de quiz IA")
    
#     # Choix du modèle Ollama
#     model = st.selectbox("Modèle Ollama :", ["mistral", "llama3.2"], index=0)
    
#     # Utilisez un champ de texte pour le sujet avec une valeur par défaut
#     topic = st.text_input("Sujet du QCM :", "Insuffisance Rénale Chronique")
    
#     # Utilisez un nombre_input au lieu d'un slider pour le nombre de questions
#     number = st.number_input(
#         "Nombre de questions :", 
#         min_value=1, 
#         max_value=20, 
#         value=5,
#         step=1  # Incréments de 1
#     )
    
#     # Sélecteur de difficulté
#     difficulty = st.selectbox("Niveau de difficulté :", ["Facile", "Moyen", "Difficile"])
    
#     # Bouton de génération du quiz
#     if st.button("Générer le QCM", use_container_width=True):
#         with st.spinner("Génération en cours..."):
#             quiz = generate_mcq(topic, number, difficulty, model)
        
#         if isinstance(quiz, dict) and "error" in quiz:
#             st.error(f"Erreur : {quiz['error']}")
#         else:
#             st.session_state["quiz_data"] = quiz
#             st.session_state["quiz_submitted"] = False
#             st.session_state["user_answers"] = {}
#             st.rerun()

# # Contenu principal
# if st.session_state["quiz_data"]:
#     # Afficher le quiz interactif
#     display_interactive_quiz(st.session_state["quiz_data"])
    
#     # Si le quiz a été soumis, afficher les résultats
#     if st.session_state["quiz_submitted"]:
#         display_quiz_results(st.session_state["quiz_data"])
# else:
#     # Message d'accueil si aucun quiz n'a encore été généré
#     st.info("👈 Utilisez le panneau de configuration à gauche pour générer un nouveau QCM médical.")
    
#     # Afficher un exemple de requête Ollama
#     with st.expander("⚙️ Exemple d'utilisation d'Ollama"):
#         st.code("""
# import requests
# response = requests.post('http://localhost:11434/api/generate', 
#                        json={
#                            "model": "mistral",
#                            "prompt": "Expliquez l'insuffisance rénale chronique"
#                        })
# print(response.json()['response'])
#         """, language="python")

# # Pied de page
# st.markdown("<div class='footer'>© 2025 AKA CARE - Tous droits réservés</div>", unsafe_allow_html=True)



import streamlit as st
import requests
import re
import json
import time
from langchain.prompts import PromptTemplate

# ✅ Initialisation des variables de session
if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = None
if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False
if "user_answers" not in st.session_state:
    st.session_state["user_answers"] = {}
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "quiz_count" not in st.session_state:
    st.session_state["quiz_count"] = 0
if "current_difficulty" not in st.session_state:
    st.session_state["current_difficulty"] = "Moyen"

# ✅ Configuration de l'API Groq
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Remplace par ta clé API

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def query_groq(prompt, model="mistral-saba-24b", max_retries=3, wait_time=5):
    """
    Envoie une requête à l'API Groq et gère les erreurs.
    """
    payload = {
        "model": model,  
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024  # Augmentez cette valeur
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=30)
            
            if response.status_code == 404:
                st.error("❌ Erreur 404 : Vérifiez l'URL de l'API Groq")
                return None

            if response.status_code == 401:
                st.error("❌ Erreur 401 : Vérifiez votre clé API Groq")
                return None

            if response.status_code == 503:
                st.warning(f"🔄 Serveur occupé, attente de {wait_time} secondes...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            
            response_json = response.json()
            if "choices" in response_json:
                return response_json["choices"][0]["message"]["content"]
            else:
                st.error(f"⚠️ Réponse inattendue de Groq : {response_json}")
                return None
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Tentative {attempt+1}/{max_retries} échouée : {e}")
            time.sleep(wait_time)

    return None

def create_prompt_with_langchain(topic, number, difficulty):
    """
    Génère un prompt optimisé pour Groq en utilisant LangChain.
    """
    template = PromptTemplate.from_template("""
    Vous êtes un expert médical en néphrologie.
    Générez EXACTEMENT {number} questions (ni plus, ni moins) de niveau {difficulty} sur le sujet suivant : "{topic}".

    **TRÈS IMPORTANT**: 
    1. Votre réponse doit contenir EXACTEMENT {number} questions numérotées de 1 à {number}.
    2. Votre réponse doit être UNIQUEMENT un objet JSON valide, sans aucun texte avant ou après.
    3. Ne commencez pas par ```json ou ``` et ne terminez pas par ```.
    
    Format exact attendu avec {number} questions:
    
    {{
        "1": {{
            "question": "Question 1 ?",
            "options": {{
                "a": "Option A",
                "b": "Option B",
                "c": "Option C",
                "d": "Option D"
            }},
            "correct": "a",
            "explanation": "Explication détaillée."
        }},
        ... répétez pour toutes les {number} questions ...
        "{number}": {{
            "question": "Question {number} ?",
            "options": {{
                "a": "Option A",
                "b": "Option B",
                "c": "Option C",
                "d": "Option D"
            }},
            "correct": "c",
            "explanation": "Explication détaillée."
        }}
    }}
    """)
    
    return template.format(topic=topic, number=number, difficulty=difficulty)

def extract_json_from_text(text):
    """
    Extrait proprement le JSON de la réponse du modèle.
    """
    if not text:
        return {"error": "Aucune réponse obtenue du modèle"}
    
    # Afficher le texte brut pour débogage (temporairement)
    st.code(text, language="json")
    
    # Tentative d'extraction et nettoyage du JSON
    try:
        # Essayer d'abord de charger le texte directement
        return json.loads(text)
    except json.JSONDecodeError:
        # Si échec, tenter d'extraire avec regex et nettoyer
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(0)
            
            # Nettoyage supplémentaire du JSON
            try:
                # Remplacer les sauts de ligne non échappés dans les chaînes
                cleaned_json = re.sub(r'(?<!\\)\\n', '\\\\n', json_text)
                # Corriger les virgules en trop avant les accolades fermantes
                cleaned_json = re.sub(r',(\s*})', r'\1', cleaned_json)
                # Ajouter les accolades manquantes si nécessaire
                if cleaned_json.count('{') > cleaned_json.count('}'):
                    cleaned_json += "}" * (cleaned_json.count('{') - cleaned_json.count('}'))
                
                return json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                st.error(f"🚨 Erreur JSON après nettoyage : {e}")
                st.code(cleaned_json, language="json")
                return {"error": f"JSON invalide : {e}"}
        else:
            return {"error": "Aucun JSON détecté dans la réponse"}

def adjust_difficulty():
    """
    Ajuste dynamiquement la difficulté en fonction du score.
    """
    score = st.session_state["score"]
    quiz_count = st.session_state["quiz_count"]

    if quiz_count < 3:
        return "Moyen"

    if score / quiz_count >= 0.8:  
        return "Difficile"
    elif score / quiz_count <= 0.4:  
        return "Facile"
    
    return "Moyen"

def generate_mcq(topic, number=5, model="mistral"):
    difficulty = adjust_difficulty()
    prompt = create_prompt_with_langchain(topic, number, difficulty)
    
    st.write("Prompt envoyé à l'API:", prompt)  # Débogage
    
    response_text = query_groq(prompt, model)
    
    st.write("Réponse brute de l'API:", response_text)  # Débogage
    
    quiz_data = extract_json_from_text(response_text)
    
    st.write("Données JSON extraites:", quiz_data)  # Débogage
    
    st.session_state["current_difficulty"] = difficulty
    return quiz_data

def display_interactive_quiz(quiz_data):
    """
    Affiche un quiz interactif avec des boutons radio pour chaque question
    et permet de soumettre les réponses pour évaluation.
    """
    if not quiz_data or "error" in quiz_data:
        st.error(f"Erreur: {quiz_data.get('error', 'Données de quiz invalides')}")
        return
    
    # Initialiser les réponses de l'utilisateur si ce n'est pas déjà fait
    if "user_answers" not in st.session_state:
        st.session_state["user_answers"] = {}
    
    # Initialiser l'état de soumission
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    
    # Afficher chaque question avec des boutons radio
    for q_num, q_data in quiz_data.items():
        # st.markdown(f"### Question {q_num}")
        
        # Afficher la question
        question_text = q_data.get("question", q_data.get("mcq", "Question manquante"))
        st.markdown(f"**{question_text}**")
        
        # Afficher les options comme boutons radio
        options = q_data.get("options", {})
        option_keys = list(options.keys())
        option_values = [f"{key}: {options[key]}" for key in option_keys]
        
        # Créer un identifiant unique pour chaque question
        q_id = f"q_{q_num}"
        
        # Afficher les boutons radio
        user_choice = st.radio(
            "Sélectionnez votre réponse:",
            option_values,
            key=q_id,
            index=None,
            disabled=st.session_state["quiz_submitted"]
        )
        
        # Enregistrer la réponse de l'utilisateur
        if user_choice:
            selected_key = user_choice.split(":")[0].strip()
            st.session_state["user_answers"][q_num] = selected_key
            
        st.divider()
    
    # Bouton de soumission
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state["quiz_submitted"]:
            if st.button("Soumettre vos réponses", 
                         use_container_width=True, 
                         disabled=len(st.session_state["user_answers"]) < len(quiz_data)):
                st.session_state["quiz_submitted"] = True
                st.rerun()
        else:
            if st.button("Réinitialiser le quiz", use_container_width=True):
                st.session_state["quiz_submitted"] = False
                st.session_state["user_answers"] = {}
                st.rerun()

def display_quiz_results(quiz_data):
    """
    Affiche les résultats du quiz avec les réponses correctes et les explications.
    """
    if not st.session_state["quiz_submitted"]:
        return
    
    st.markdown("## 📊 Résultats du Quiz")
    
    correct_count = 0
    total_questions = len(quiz_data)
    
    incorrect_answers = []  # Pour stocker les questions mal répondues
    
    for q_num, q_data in quiz_data.items():
        user_answer = st.session_state["user_answers"].get(q_num, "")
        correct_answer = q_data.get("correct", "")
        
        # Affichage de la question
        question_text = q_data.get("question", q_data.get("mcq", ""))
        st.markdown(f"**{q_num}. {question_text}**")
        
        # Vérifier si la réponse est correcte
        is_correct = user_answer == correct_answer
        if is_correct:
            correct_count += 1
        else:
            incorrect_answers.append(q_num)
        
        # ✅ Afficher la réponse de l'utilisateur
        st.markdown("### 📝 Vos réponses")
        options = q_data.get("options", {})
        
        for opt_key, opt_value in options.items():
            prefix = "✅ " if opt_key == correct_answer else "❌ " if opt_key == user_answer else "   "
            highlight = "**" if opt_key == user_answer else ""
            st.markdown(f"{prefix} {highlight}{opt_key}: {opt_value}{highlight}")

        # ✅ Afficher le bon choix
        if not is_correct:
            st.warning(f"💡 La bonne réponse était : **{correct_answer} - {options.get(correct_answer, '')}**")

        # ✅ Afficher l'explication
        explanation_key = "explanation" if "explanation" in q_data else "explication"
        if explanation_key in q_data:
            st.info(f"📚 **Explication**: {q_data[explanation_key]}")
        
        st.divider()
        
        st.session_state["score"] += correct_count
        st.session_state["quiz_count"] += 1
        
    
    # ✅ Afficher le score global
    score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"## 🏆 Score final: **{correct_count}/{total_questions}** ({score_percentage:.1f}%)")

        # ✅ Feedback basé sur le score
        if score_percentage >= 80:
            st.balloons()
            st.success("🎉 Excellent ! Vous maîtrisez bien ce sujet !")
        elif score_percentage >= 60:
            st.success("👍 Bon travail ! Vous avez une bonne compréhension du sujet.")
        elif score_percentage >= 40:
            st.warning("🔍 Pas mal, mais il y a encore des points à améliorer.")
        else:
            st.error("📚 Ce sujet nécessite plus de révision. Ne vous découragez pas !")

    # # ✅ Affichage des erreurs
    # if incorrect_answers:
    #     st.markdown("### ❌ Questions mal répondues")
    #     for q_num in incorrect_answers:
    #         st.markdown(f"- **Question {q_num}**")

st.markdown("""
    <style>
        /* 💎 Background et design */
        body {
            background-color: #FBF8F1;
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            color: #EF4D89;
            font-size: 2.5rem;
            font-weight: bold;
            padding: 1rem 0;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #00BAC6;
            font-size: 1.5rem;
            margin-bottom: 2rem;
            border-top: 2px solid #00BAC6;
        }
        .stButton>button {
            background-color: #EF4D89 !important;
            color: white !important;
            font-size: 1.1rem !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        .stButton>button:hover {
            background-color: #D43F76 !important;
        }
        .stRadio > label {
            font-size: 1rem;
            font-weight: bold;
            color: #34495E;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            # border-top: 2px solid #00BAC6;
            font-size: 0.9rem;
            color: #95A5A6;
        }
    </style>
""", unsafe_allow_html=True)

# 🚀 **Interface Streamlit**
st.markdown("<h1 class='main-header'>AKA CARE</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>Quiz Éducatif sur les Maladies Rénales Chroniques</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Générateur de quiz IA")
    model = st.selectbox("Modèle Groq :", ["mistral-saba-24b", "llama3-70b-8192", "llama3"], index=0)
    topic = st.text_input("Sujet du QCM :", "Insuffisance Rénale Chronique")
    number = st.number_input("Nombre de questions :", min_value=0, max_value=20, value=5, step=1)

    if st.button("Générer le QCM", use_container_width=True):
        with st.spinner("Génération en cours..."):
            quiz = generate_mcq(topic, number, model)
        
        if isinstance(quiz, dict) and "error" in quiz:
            st.error(f"Erreur : {quiz['error']}")
        else:
            st.session_state["quiz_data"] = quiz
            st.session_state["quiz_submitted"] = False
            st.session_state["user_answers"] = {}
            st.rerun()

if st.session_state["quiz_data"]:
    st.markdown(f"### 📝 Niveau actuel : **{st.session_state['current_difficulty']}**")

    # Afficher le quiz interactif
    display_interactive_quiz(st.session_state["quiz_data"])
    
    # Si le quiz a été soumis, afficher les résultats
    if st.session_state["quiz_submitted"]:
        display_quiz_results(st.session_state["quiz_data"])
        
