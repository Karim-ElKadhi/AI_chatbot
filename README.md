# AI Chatbot Platform for Insurance Services

MediBot is an intelligent conversational platform built with Python. It combines advanced features including:

- 🎙️ Real-time **voice transcription**
- 💬 **Emotion and intent detection** from user messages
- 📊 **Graphical emotion tracking** throughout conversations
- 🔐 Secure **user/admin authentication** 
- 🗃️ Persistent **conversation history storage** in SQLite
- ⭐ **Automatic satisfaction scoring** based on emotion analysis

---

## 🚀 Key Features

### 🗣️ Multimodal Interaction
- Accepts user input via **text or voice** .


### 🧠 Emotion Analysis
- Detects emotion in each message (e.g., *positive, angry, confused, calm*).
- Displays a **dynamic emotion graph** .

### 🔐 Authentication
- Supports JWT-based authentication for:
  - **Admin users**: access dashboard and monitoring
  - **Regular users**: access conversation interface

### 💾 SQLite Database
- Conversations are stored in a database.
### 📈 Admin Dashboard
- Admins can view all conversations and message-level details.
- Each conversation includes a **satisfaction score** based on detected emotions.


---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/<your_username>/AI-chatbot.git
cd AI-chatbot
pip install -r requirements.txt
```
📄 License

This project is licensed under the MIT License.
See the file LICENSE
 for more details.
