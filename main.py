import streamlit as st
import requests
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from audio_recorder_streamlit import audio_recorder
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langfuse import Langfuse
import uuid
import os
import sqlite3
import json
from datetime import datetime
import uuid
import altair as alt
import bcrypt
import google.generativeai as genai
import re

# Importer les textes multilingues a partir du fichier lang.py
from lang import texts

#cle open ai 
openai.api_key = "Insert_your_openai_api_key_here" 
# Configuration
GROQ_API_KEY = "Insert_your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
genai.configure(api_key="Insert_your_google_api_key_here")

#import torch
#print(torch.cuda.is_available())
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(0))
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# Chargement du modèle Embedding
bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')


# Fichier CSV contenant les instructions / réponses
file_path = "/dataset/bitext-insurance-llm-chatbot-training-dataset.csv"
df = pd.read_csv(file_path)

# Connexion ou création
conn = sqlite3.connect("Database/conversations.db")
cursor = conn.cursor()



# Initialisation session
if "page" not in st.session_state:
    st.session_state.page = "home"
if "conversation_saved" not in st.session_state:
    st.session_state.conversation_saved = False
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role_user = None    
for key in ["chat_history", "user_input", "send"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else "" if key == "user_input" else False
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []
    if "intent_history" not in st.session_state:
        st.session_state.intent_history = []    

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 180px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 180px;
        }
        .css-1d391kg { 
            padding: 1rem 0.5rem;}
    </style>
""", unsafe_allow_html=True)


# --- Configuration initiale de la langue ---
if "lang" not in st.session_state:
    st.session_state.lang = "fr"  # Valeur par défaut

# Mapping langue -> label avec emoji
language_options = {
    "fr": "Français",
    "en": "English",
    "ar": "العربية"
}

language_labels = list(language_options.values())
language_keys = list(language_options.keys())
current_index = language_keys.index(st.session_state.get("lang", "fr"))



# --- Affichage du selectbox avec les drapeaux ---
with st.container():
    selected_label = st.selectbox(" ", language_labels, index=current_index, key="language_selectbox")

# --- Mise à jour de la langue en session ---
st.session_state.lang = language_keys[language_labels.index(selected_label)]
lang = st.session_state.lang  


# Saisie utilisateur
def display_response(response: str, intent: str = None) -> str:
    if intent:
        return f" {response}"
    else:
        return f" {response}"


#Fonction pour le chatbot administrateur
def generate_and_execute_admin_prompt(prompt: str):
    system_prompt = (
    "You are an analytics assistant dedicated to helping an admin user analyze customer conversations. "
    "You must only generate Python code that performs:\n"
     "- Aggregated KPI calculations (e.g., average satisfaction, number of conversations per user)\n"
    "- Visualizations using Altair charts\n"
    "- Behavioral insights about individual users (e.g., customer Karim's satisfaction trend, attention needed if satisfaction below 6.0)\n"
    "- Strategic insights and actionable recommendations for decision-makers based on the data patterns\n\n"

    "You must only respond with Python or SQL code based strictly on the following data: "
    "The only available data is a SQLite table named `conversations` with the columns:\n"
    "- id (int)\n"
    "- timestamp (datetime string)\n"
    "- prompt_count (int)\n"
    "- content (JSON string)\n"
    "- satisfaction_score (float from 0 to 10)\n"
    "- username (text)\n\n"
                    " Generate only Python code using pandas and Altair (do NOT use matplotlib, seaborn, or any other libs). "
     "You may use `mark_arc()` in Altair to create pie/donut charts if the request asks for it."
        "     You may use the following Altair chart types:"
        "- mark_bar() for bar charts"
        "- mark_line() for time-series"
        "- mark_area() for cumulative data"
        "- mark_circle() or mark_point() for scatter plots"
        "- mark_boxplot() for distribution"
    
    "You must only use the existing SQLite connection 'conn' (do NOT use sqlite3.connect()). "
    "If you use a date field (like from strftime), make sure to convert it using `pd.to_datetime()` before plotting in Altair. "
    "Your task is to generate Python code that performs analytics or creates visualizations based solely on this database. "
    "You must not invent any data or information. "
    "You must generate Python code that calculates KPIs and assigns the plain text result to a variable named `result` (e.g., result = 'Average satisfaction: 8.2')."
    "Always return your code enclosed in a ```python``` code block, and nothing else outside of that block."
    "You must assign your textual analysis or results to a variable called `result`, for example:\n"
    "result = 'The user Karim has a satisfaction average of 6.2 and needs more attention.'\n"
    "You may also assign a variable `recommendation` when patterns suggest clear actions (e.g., low satisfaction, decreasing usage). "
    "If no clear recommendation can be drawn, omit it.\n"
     "If the request is unrelated to KPIs or visualizations (e.g., jokes, contract questions), respond:\n"
    "\"Sorry, I can only generate KPIs or charts based on the available database.\"\n"
            "Only output text outside of the code block when denying a request."

)


    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=data)
    if resp.status_code != 200:
        return "Erreur avec le modèle LLM", None

    content = resp.json()["choices"][0]["message"]["content"]

    # Extraire le code entre ```python ... ```
    code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    if not code_match:
        return content, None

    code = code_match.group(1)

    # Environnement sécurisé avec uniquement certaines libs
    local_vars = {"pd": pd, "conn": conn, "alt": alt,"st":st}
    try:
        exec(code, {}, local_vars)
        # On cherche le graphique généré (type Altair Chart)
        chart = next((v for v in local_vars.values() if hasattr(v, 'mark_line') or hasattr(v, 'mark_bar')), None)
        # Cas KPI : résultat dans une variable 'result' définie dans le code
        result = local_vars.get("result", None)
        return result if result else content.split("```")[0].strip(), chart
    except Exception as e:
        return f"Erreur d'exécution : {e}", None

# Fonction pour afficher les graphiques et kpis pour l'assistant administrateur
def show_admin_analytics_bot():
    st.title("📊 Assistant Administratif Intelligent")

    admin_prompt = st.text_input("🧠",key="admin_prompt_input",placeholder="Votre requête" )
    
    if st.button("Analyser"):
        if admin_prompt.strip():
            with st.spinner("Analyse en cours..."):
                response_text, chart = generate_and_execute_admin_prompt(admin_prompt)
                st.markdown("### Réponse :")
                st.write(response_text)
                print(response_text)
                
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                            
        else:
            st.warning("❗ Veuillez entrer une requête.")

# Fonction pour détecter la langue du texte
def detect_language(text: str) -> str:
    prompt = f"Detect the language of this sentence: \"{text}\"\nReply only with: fr, en, or ar."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content'].strip().lower() if response.status_code == 200 else "en"

# Fonction pour obtenir le prompt système pour le client en fonction de la langue
def get_system_prompt(language: str) -> str:
    if language == "fr":
        return (
            "Vous êtes un assistant virtuel professionnel pour une compagnie d'assurance. "
            "Vous ne devez répondre qu'aux questions liées à l'assurance (contrats, garanties, sinistres, résiliations, etc.). "
             "- Cependant, si l'utilisateur vous dit bonjour, au revoir ou vous remercie meme s'il l'a ecrit incorrectement, répondez poliment dans la même langue.\n"
            "Si la question ne concerne pas l'assurance, dites poliment : "
            "'Je suis désolé, je ne peux répondre qu'à des questions relatives au domaine de l'assurance.' "
            "Chaque réponse doit inclure au moins une recommandation pour le client ."
        )
    elif language == "en":
        return (
            "You are a professional virtual assistant for an insurance company. "
            "Only respond to questions strictly related to insurance even if he misspell it. "
            "- However, if the user says hello, goodbye, or thanks , respond politely in the same language.\n"
            "If the user says ok,thank you, respond: You're welcome. Is there anything I can help you with regarding insurance?"
            "If the user asks something unrelated, respond: "

            "'I'm sorry, I can only help with insurance-related questions.' "
            "- Consider the CONTEXT of previous user messages to give relevant recommendations."

        )
    elif language == "ar":
        return (
            "أنت مساعد افتراضي محترف لشركة تأمين. "
            "يجب أن تجيب فقط على الأسئلة المتعلقة بالتأمين مثل العقود، المطالبات، الأسعار، الإلغاء. "
            "- ومع ذلك، إذا قال لك المستخدم مرحبًا أو وداعًا أو شكرًا، فقم بالرد بأدب بنفس اللغة.\n"
            "إذا طرح المستخدم سؤالاً غير متعلق بالتأمين، فقل: "
            "'عذرًا، يمكنني فقط مساعدتك في الأسئلة المتعلقة بالتأمين.' "
            "يجب أن تتضمن كل إجابة توصية عملية واحدة على الأقل يمكن للعميل تنفيذها."
        )
    else:
        return get_system_prompt("en") 

# Fonction pour trouver la meilleure correspondance dans le dataset en utilisant la similarité sémantique
def find_best_match(prompt: str, df: pd.DataFrame):

    # Sémantique
    instruction_embeddings = bert_model.encode(df['instruction'].astype(str).tolist(), convert_to_tensor=True)
    input_embedding = bert_model.encode(prompt, convert_to_tensor=True)
    similarities_semantic = cosine_similarity(
        [input_embedding.cpu().numpy()],
        instruction_embeddings.cpu().numpy()
    ).flatten()
    best_match_index_semantic = similarities_semantic.argmax()
    best_match_score_semantic = similarities_semantic[best_match_index_semantic]

    if best_match_score_semantic > 0.65:
        return df.iloc[best_match_index_semantic]['response'], df.iloc[best_match_index_semantic]['intent']
    else:
        return None, None

# Fonction pour obtenir la réponse du chatbot d'assurance coté client
def get_insurance_response(prompt: str, df: pd.DataFrame):
    # 1. Détection de la langue utilisateur
    language = detect_language(prompt)

    # 2. Traduction du prompt vers l'anglais si nécessaire (pour matching)
    prompt_for_matching = prompt
    if language != "en":
        prompt_for_matching = translate_to_english(prompt)

    # 3. Génération de la réponse

    fallback = get_insurance_response(prompt, language)
    return display_response(fallback), fallback


# Fonction pour traduire le texte en anglais
def translate_to_english(text: str) -> str:
    prompt = (
        "Translate the following message into English. "
        "Do not alter the meaning or structure.\n\n"
        f"{text}"
    )

    try:
        model = genai.GenerativeModel("gemma-3n-e4b-it")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR - Gemini Translate to English]: {e}")
        return text


# Fonction pour traduire la réponse a partir du dataset dans la langue cible
def translate_response(text: str, target_lang: str) -> str:
    if target_lang == "en":  
        return text

    lang_map = {"fr": "French", "ar": "Arabic"}
    prompt = (
        f"You are a professional insurance translator.\n"
        f"Translate the following response into {lang_map[target_lang]}, but DO NOT translate or modify the placeholders inside double curly braces (like {{WEBSITE_URL}} or {{MY_POLICIES_SECTION}} , etc).\n"
        f"Preserve all formatting, line breaks, numbers, and bullet points.\n"
        f"Do not change the structure or remove anything.\n"
        f"---\n{text}\n---"
    )

    try:
        model = genai.GenerativeModel("gemma-3n-e4b-it")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR - Gemini Translation]: {e}")
        return text


# Fonction pour détecter le sentiment de l'utilisateur
def detect_sentiment(user_msg: str) -> str:
    prompt = f"""
You are an emotion analysis model. Here is a user's message:

"{user_msg}"

Classify the dominant emotion from the following list: frustrated,positive, neutral, confused, 
stressed, angry, calm ,convinced,reassured,concerned.

Respond with only one word (the emotion).
"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        emotion = response.json()['choices'][0]['message']['content'].strip().lower().strip('.')
        return emotion
    else:
        return "undetected"

def generate_title_from_llm(prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "Conversation sans titre"



# Liste des émotions et leur index
emotion_labels = [
    "angry",         # worst
    "frustrated",
    "stressed",
    "confused",
    "neutral",       # middle
    "calm",
    "reassured",
    "convinced",
    "positive"       # best
]
emotion_to_index = {e: i for i, e in enumerate(emotion_labels)}


# Fonction pour hasher le mot de passe
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Fonction pour vérifier le mot de passe
def check_password(password, password_hash):
    return bcrypt.checkpw(password.encode(), password_hash.encode())

# Fonction pour authentifier l'utilisateur
def authenticate_user(username, password):
    cursor.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if result and check_password(password, result[0]):
        return result[1]  # Retourne le rôle
    return None


# Conversion des lignes du dataset en documents utilisables par LangChain
docs = [
    Document(
        page_content=f"Question: {row['question']}\nRéponse: {row['response']}",
        metadata={
            "intent": row["intent"],
            "category": row["category"]
        }
    )
    for _, row in df.iterrows()
]


# Création du magasin vectoriel FAISS à partir des documents encodés
vectorstore = FAISS.from_documents(docs, bert_model)


# Sauvegarde locale pour réutilisation sans recalculer les embeddings :
vectorstore.save_local("insurance_faiss_index")

# Création du retriever pour obtenir les passages les plus proches
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialisation du modèle de génération Groq
llm = ChatGroq(
    temperature=0.7,  # contrôle de la créativité de la réponse
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=GROQ_MODEL  
)
# Définition du prompt utilisé dans la chaîne RAG 

template = """
{system_prompt}

Use the following context to answer the user's question as clearly and accurately as possible and adapt to the specified language.

Contexte :
{context}

Question :
{question}

Réponse :
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["system_prompt", "context", "question"]
)
# Création de la chaîne RAG (retrieval + génération)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)
# Fonction principale pour interagir avec le chatbot

def get_insurance_response(prompt_utilisateur: str, langue: str = "fr"):
    trace = langfuse.trace(name="insurance_rag_chat")
    trace.update_input({"user_prompt": prompt_utilisateur, "language": langue})

    # Création du prompt personnalisé
    system_prompt = get_system_prompt(langue)

    reponse = rag_chain.run({
        "system_prompt": system_prompt,
        "question": prompt_utilisateur
    })
    #utilisation du tracer de langfuse pour Tracer les interactions utilisateur-LLM , Analyser la qualité des réponses , Analyser la qualité des réponses (Temps de réponse ,Nombre de tokens utilisés,Latence du modèle)
    trace.update_output({"response": reponse})
    trace.end()
    return reponse

# Fonction pour détecter l'intention de l'utilisateur pour les reponses non trouvées dans le dataset
def detect_intention(user_msg: str) -> str:
    prompt = f"""
You are an intent classifier for an insurance chatbot.
Classify the intent of this message:

"{user_msg}"

Choose one: ask_quote, cancel_policy, request_info, make_complaint ,etc.
else if the intent is not in the list, respond with general intent like greeting, goodbye, thanks, etc.

Respond with only one word.
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content'].strip().lower() if response.status_code == 200 else "undetected"


def show_user_conversation_history(conn):

    username = st.session_state.username


    # Charger toutes les conversations de l'utilisateur
    query = "SELECT timestamp, titre, instruction, reponse FROM usercv WHERE username = ? ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn, params=(username,))

    if df.empty:
        st.info("ℹ️ Vous n'avez encore enregistré aucune conversation.")
        return

    # Affichage sous forme d'expander
    for _, row in df.iterrows():
        title = row["titre"]
        date  = row["timestamp"].split("T")[0] 
        conv_data = json.loads(row["instruction"])

        with st.expander(f"🗂️ {title} — {date}"):
            try:
                instructions = json.loads(row["instruction"])
                reponses = json.loads(row["reponse"])
            except json.JSONDecodeError:
                st.error("❌ Erreur de lecture du contenu de la conversation.")
                continue
            # Afficher les paires instruction → réponse
            for i in range(min(len(instructions), len(reponses))):
                st.markdown(f"**👤 Vous :** {instructions[i]}")
                st.markdown(f"**🤖 Bot :** {reponses[i]}")
                st.markdown("---")
    
def on_enter_pressed():
    st.session_state.send = True

# Fonction pour faire parler le bot avec OpenAI TTS
def speak(text, voice="nova"):
    audio_path = f"tts_{uuid.uuid4()}.mp3"

    response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    response.stream_to_file(audio_path)
    return audio_path


# Fonction pour sauvegarder la conversation dans dans la base de données conversations.db
def save_conversation_sqlite():
    prompts = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    emotions = st.session_state.emotion_history
    intentions = st.session_state.intent_history
    emotion_score_map = {
        "angry": 0,
        "frustrated": 1,
        "stressed": 2,
        "confused": 3,
        "neutral": 5,
        "calm": 6,
        "convinced": 7,
        "reassured": 8,
        "positive": 10
    }

    # Calcul score moyen
    scores = [emotion_score_map.get(e, 5) for e in emotions]  # par défaut 5 (neutre)
    satisfaction_score = round(sum(scores) / len(scores), 2) if scores else 5.0

    content_dict = {}
    for i, prompt in enumerate(prompts):
        content_dict[prompt] = {
            "sentiment": emotions[i] if i < len(emotions) else None,
            "intention": intentions[i] if i < len(intentions) else None
        }
    prompts_summary = prompts[:3]

    llm_prompt = (
        "You are an assistant tasked with generating a short title (max 5 words) "
        "that summarizes the topic of a customer conversation for an insurance chatbot in the language of the conversation (english , arabic or french). "
        "Here are the first 3 user prompts:\n\n" +
        "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts_summary)]) +
        "\n\nReturn only the title."
    )
    title = generate_title_from_llm(llm_prompt)
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    prompt_count = len(prompts)
    content_json = json.dumps(content_dict, ensure_ascii=False)
    username = st.session_state.username
    # Insertion dans la base
    cursor.execute("""
        INSERT INTO conversations (id, timestamp, prompt_count, content, satisfaction_score, username,titre)
        VALUES (?, ?, ?, ?, ?, ?,?)
    """, (conversation_id, timestamp, prompt_count, content_json, satisfaction_score, username,title))
    conn.commit()

def save_conversation_user():
    instruction = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    reponse = [msg["content"] for msg in st.session_state.chat_history if msg["role"] != "user"]
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    content_in = json.dumps(instruction, ensure_ascii=False)
    content_rep = json.dumps(reponse, ensure_ascii=False)

    prompts_summary = instruction[:3]

    llm_prompt = (
        "You are an assistant tasked with generating a short title (max 5 words) "
        "that summarizes the topic of a customer conversation for an insurance chatbot in the language of the conversation (english , arabic or french). "
        "Here are the first 3 user prompts:\n\n" +
        "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts_summary)]) +
        "\n\nReturn only the title."
    )
    title = generate_title_from_llm(llm_prompt)
    username = st.session_state.username
    # Insertion dans la base
    cursor.execute("""
        INSERT INTO usercv (id, timestamp, instruction, reponse, username,titre)
        VALUES (?, ?, ?, ?, ?,?)
    """, (conversation_id, timestamp,content_in ,content_rep , username,title))
    conn.commit()
# afficher le graphique des émotions pour l'administrateur
def build_emotion_chart():
    prompts = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    emotions = st.session_state.emotion_history

    if not prompts or not emotions:
        st.warning("No emotion data available.")
        return

    data = []
    for i, (prompt, emotion) in enumerate(zip(prompts, emotions), start=1):
        if emotion in emotion_to_index:
            data.append({
                "Prompt ID": f"P{i}",  
                "Prompt Text": prompt,
                "Emotion Index": emotion_to_index[emotion],
                "Emotion Label": emotion
            })

    df = pd.DataFrame(data)
    return df

# === Page d'accueil ===
if st.session_state.page == "home":
    st.title(texts["main_title"][lang])
    if not st.session_state.authenticated:
        st.title("🔐 Connexion")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            role_user = authenticate_user(username, password)
            if role_user:
                st.session_state.authenticated = True
                st.session_state.role_user = role_user
                st.session_state.username = username
                st.success(f"Connecté en tant que {username}")
                st.session_state.page = "chat"
            else:
                st.error("Identifiants incorrects")
        st.stop()
    


# INTERFACE À ONGLET
elif st.session_state.page == "chat":
    
    tabs = [texts["tab_title"][lang]]
    if st.session_state.role_user == "admin":
        tabs.append("📉 Suivi émotionnel")
        tabs.append("📁 Historique des conversations")
        tabs.append("👥 Gestion Utilisateurs")
    else:
        tabs.append(texts["hist"][lang])

    # ===== CHATBOT TAB =====
    tab_objects = st.tabs(tabs)
        # === Bouton de déconnexion ===
    with st.sidebar.expander("🔍Info", expanded=True):
        st.title("👤 Utilisateur")
        st.write(f"Connecté en tant que : `{st.session_state.username}`")
        if st.button("Déconnecter du compte"):
            # Réinitialisation des variables de session
            if not st.session_state.conversation_saved:
                    save_conversation_sqlite()
                    save_conversation_user()
                    st.session_state.conversation_saved = True
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role_user = None
            st.session_state.page = "home"
            # Réinitialisation partielle de la session si besoin
            st.session_state.chat_history = []
            st.session_state.emotion_history = []
            st.session_state.intent_history = []
            st.session_state.conversation_saved = False
            st.rerun()
    with tab_objects[0]:
        if st.session_state.role_user == "admin":
            show_admin_analytics_bot()
        
        else:
        # Titre
            if not st.session_state.conversation_saved and st.button(texts["quit"][lang]):
                # Sauvegarder si pas encore fait
                if not st.session_state.conversation_saved:
                    save_conversation_sqlite()
                    save_conversation_user()
                    st.session_state.conversation_saved = True

                # Réinitialiser la conversation mais rester sur la page "chat"
                st.session_state.chat_history = []
                st.session_state.emotion_history = []
                st.session_state.intent_history = []
                st.session_state.send = False
                st.session_state.send_from_voice = False
                st.session_state.conversation_saved = False
                st.rerun()
            st.title(texts["main_title"][lang])

            # Affichage de la conversation
            user_index = 0

            for i, msg in enumerate(st.session_state.chat_history):
                role = texts["role_labels"][lang]["user"] if msg["role"] == "user" else texts["role_labels"][lang]["bot"]
                st.markdown(f"**{role}:** {msg['content']}")

                # Only increment user_index if it's a user message
                if msg["role"] == "user":
                    if user_index < len(st.session_state.emotion_history):
                        emotion = st.session_state.emotion_history[user_index]
                    user_index += 1
                
                # Si c'est un message du bot, ajouter le bouton "Écouter"
                if msg["role"] != "user":
                    if st.button(texts["listen"][lang], key=f"listen_{i}"):
                        text_to_speak = msg["content"]
                        audio_path = speak(text_to_speak)  # Convert to voice
                        st.audio(audio_path, format="audio/mp3")
                        
                        os.remove(audio_path)


            # Traitement message texte
            if st.session_state.send and st.session_state.user_input.strip():
                user_msg = st.session_state.user_input.strip()
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                emotion = detect_sentiment(translate_to_english(user_msg))
                st.session_state.emotion_history.append(emotion)
                intent = find_best_match(translate_to_english(user_msg), df)[1]
                if not intent:
                    intent = detect_intention(translate_to_english(user_msg))
                bot_reply, bot_reply_raw = get_insurance_response(user_msg, df)
                st.session_state.intent_history.append(intent)    
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                st.session_state.user_input = ""
                st.session_state.send = False
                st.rerun()
            # Traitement message vocal
            if st.session_state.get("send_from_voice", False):
                voice_msg = st.session_state.get("voice_input", "").strip()
                if voice_msg:
                    st.session_state.chat_history.append({"role": "user", "content": voice_msg})
                    emotion = detect_sentiment(translate_to_english(voice_msg))
                    st.session_state.emotion_history.append(emotion)
                    intent = find_best_match(translate_to_english(voice_msg), df)[1]
                    if not intent:
                        intent = detect_intention(translate_to_english(voice_msg))
                    bot_reply, bot_reply_raw = get_insurance_response(voice_msg, df)
                    st.session_state.intent_history.append(intent)  
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    st.session_state.send_from_voice = False
                st.session_state.voice_input = ""
                st.rerun()

            


            # Champ de texte
            st.text_input(
            texts["user_input"][lang],
            key="user_input",
            on_change=on_enter_pressed,
            placeholder=texts["placeholder"][lang]
                )
            # === 🎤 ENREGISTREMENT VOCAL
            st.markdown(texts["voice"][lang])
            audio_bytes = audio_recorder(
                key="voice_recorder",
                text="🎙️ Appuyez pour enregistrer" if lang == "fr" else (
                    "🎙️ Tap to record" if lang == "en" else "🎙️ اضغط للتسجيل"
                ),                icon_size="1.5x",
                pause_threshold=4.0
            )
            if audio_bytes:
                with open("audio_temp.wav", "wb") as f:
                    f.write(audio_bytes)

                with st.spinner(texts["transcribing_spinner"][lang]):
                    try:
                        files = {"file": open("audio_temp.wav", "rb")}
                        data = {"model": "whisper-large-v3", "language": "en"}
                        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                        response = requests.post(GROQ_AUDIO_URL, headers=headers, files=files, data=data)

                        if response.status_code == 200:
                            transcribed_text = response.json()["text"]
                            st.success(f"{texts['transcription_success'][lang]} *{transcribed_text}*")

                            st.session_state.voice_input = transcribed_text
                            st.session_state.send_from_voice = True

                            st.rerun()
                        else:
                            st.error(f"❌ Erreur Whisper API : {response.text}")

                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")


            with st.expander(texts["contact"][lang]):
                st.markdown(texts["contact_details"][lang])


    

    
    if st.session_state.role_user == "admin":    

        # ===== EMOTION TIMELINE TAB =====
        with tab_objects[1]:

            st.markdown("### 📉 Suivi émotionnel")
            emotion_score_map = {
                "angry": 0,
                "frustrated": 1,
                "stressed": 2,
                "confused": 3,
                "neutral": 5,
                "calm": 6,
                "convinced": 7,
                "reassured": 8,
                "positive": 10
            }
            # Sélecteur de date (calendrier)
            selected_date = st.date_input("📅 Choisissez une date :", format="YYYY-MM-DD")

            # Récupérer les utilisateurs distincts ayant une conversation
            cursor.execute("SELECT DISTINCT username FROM conversations")
            usernames = [row[0] for row in cursor.fetchall()]
            selected_user = st.selectbox("👤 Choisissez un utilisateur :", usernames)

            # Modifier la requête SQL existante :
            cursor.execute("""
                SELECT id, timestamp,titre, content FROM conversations 
                WHERE DATE(timestamp) = ? AND username = ?
                ORDER BY timestamp DESC
            """, (selected_date.isoformat(), selected_user))
            conversations = cursor.fetchall()

            if not conversations:
                st.info("ℹ️ Aucune conversation enregistrée pour cette date.")
            else:
                # Liste déroulante pour sélectionner une conversation spécifique
                conv_choices = [f"{timestamp.split('T')[0]} | {titre}" for conv_id,titre, timestamp, _ in conversations]
                selected_conv = st.selectbox("💬 Choisissez une conversation :", conv_choices)

                selected_conv_id = selected_conv.split("|")[1].strip()
                selected_conv_data = next((row for row in conversations if row[1] == selected_conv_id), None)

                if selected_conv_data:
                    conv_content = json.loads(selected_conv_data[3])

                    data = []
                    for i, (prompt, meta) in enumerate(conv_content.items(), start=1):
                        emotion = meta.get("sentiment", "unknown")
                        if emotion in emotion_to_index:
                            data.append({
                                "Prompt ID": f"P{i}",
                                "Prompt Text": prompt,
                                "Emotion Index": emotion_to_index[emotion],
                                "Emotion Label": emotion
                            })

                    if data:
                        df_emotions = pd.DataFrame(data)
                        emotion_order = list(emotion_score_map.keys())[::-1]
                        st.markdown("### 📊 Courbe des émotions")
                        chart = alt.Chart(df_emotions).mark_line(point=True).encode(
                            x="Prompt ID",
                            y=alt.Y("Emotion Label", sort=emotion_order),
                            tooltip=["Prompt Text", "Emotion Label"]
                        ).properties(height=400)

                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("⚠️ Aucun tag émotionnel trouvé dans cette conversation.")
    # Onglet Historique des conversations
    if st.session_state.role_user == "admin":    

        with tab_objects[2]:
            st.title("📁 Historique des conversations")
            df = pd.read_sql_query("SELECT * FROM conversations", conn, parse_dates=["timestamp"])
            selected_date = st.date_input("📅 Sélectionner une date :", value=pd.to_datetime("today"))
            
            filtered_df = df[df["timestamp"].dt.date == selected_date]
            # Convertir chaque contenu JSON en texte lisible 
            filtered_df["content"] = filtered_df["content"].apply(
                lambda c: json.dumps(json.loads(c), ensure_ascii=False, indent=2)
            )
            if filtered_df.empty:
                st.info("Aucune conversation enregistrée à cette date.")
            else:
                for _, row in filtered_df.iterrows():
                    st.markdown(f"### 💬 Titre : `{row['titre']}`")
                    st.markdown(f"🕒 Date: {row['timestamp']}")
                    st.markdown(f"🧾 Nombre de prompts : {row['prompt_count']}")
                    st.code(row['content'], language='json')
                    st.markdown(f"👤 Utilisateur: `{row['username']}`")
                    st.markdown(f"📊 Score de satisfaction : **{row['satisfaction_score']}/10**")
                    st.markdown("---")

    # Onglet Gestion des utilisateurs
    if st.session_state.role_user == "admin":    
        with tab_objects[3]:
            st.title("👥 Ajouter un utilisateur")
            new_username = st.text_input("👤 Nom d'utilisateur")
            new_password = st.text_input("🔑 Mot de passe", type="password")
            new_role = st.selectbox("🔧 Rôle", ["user", "admin"])
            
            if st.button("➕ Ajouter l'utilisateur"):
                new_username = str(new_username).strip()
                new_role = str(new_role)
                new_password = str(new_password)
                hashed_pw = hash_password(new_password)
                try:
                    cursor.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                                (new_username, hashed_pw, new_role))
                    conn.commit()
                    st.success(f"✅ Utilisateur '{new_username}' ajouté avec succès.")
                except sqlite3.IntegrityError:
                    st.error("⚠️ Ce nom d'utilisateur existe déjà.")
            st.button("🗑️ Supprimer un utilisateur")
            st.button("✏️ Modifier un utilisateur")
    with tab_objects[1]:
        if st.session_state.role_user != "admin":
            show_user_conversation_history(conn)
