# Visionary AI Chatbot

A sophisticated, multimodal AI Chatbot application built with Flask, powered by local Large Language Models (LM Studio) and advanced Retrieval-Augmented Generation (RAG). This application features a custom self-repair mechanism (Model Context Protocol), dynamic web-search fallback, and persistent local MySQL storage.

## ✨ Features
* **Privacy-First Local AI**: Powered completely by local models. Uses LM Studio for conversational generation and Ollama for vector embeddings, ensuring zero data leakage.
* **Smart Context (RAG)**: Uses FAISS to index and search through documents to provide grounded, accurate answers.
* **Contextual Memory**: Seamlessly remembers past conversations and references previous messages within the chat flow.
* **Intelligent Web Fallback**: If the local AI and document index don't know the answer, it autonomously searches the internet (via DuckDuckGo/Google) to find real-time information.
* **Self-Repairing Responses (MCP)**: Uses a standalone Model Context Protocol (MCP) server to evaluate and fix the AI's own responses before they are shown to the user.
* **MySQL Database Integration**: Stores user profiles, chat history, and conversation metadata using SQLAlchemy.

## 🛠️ Technology Stack
* **Backend**: Python, Flask, SQLAlchemy
* **Database**: MySQL
* **AI Models**: 
  * Generation: `mistral-7b-instruct-v0.2` (via LM Studio)
  * Embeddings: `nomic-embed-text` (via Ollama)
* **Vector Store**: FAISS
* **Tools**: LangChain, BeautifulSoup (Web Scraping)

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed and running on your local machine:
* Python 3.10+
* MySQL Server (e.g., via XAMPP or WAMP) running on port `3306`
* [LM Studio](https://lmstudio.ai/) running a local server on port `1234`
* [Ollama](https://ollama.com/) running a local server on port `11434`

### 2. Installation
Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/nirali0087/Chatbot_AI.git
cd Chatbot_AI
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your MySQL database credentials:

```env
DB_HOST=127.0.0.1
DB_NAME=chatbot_db
DB_USER=root
DB_PASSWORD=your_password_here
DB_PORT=3306

SECRET_KEY=your_secret_key
DEBUG=True

LM_STUDIO_BASE_URL=http://localhost:1234/v1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
LM_STUDIO_LLM_MODEL=mistral-7b-instruct-v0.2
```
*Note: Make sure to create a blank database named `chatbot_db` in your MySQL server before running the app.*

### 4. Running the Application
To enable the advanced self-repair (MCP) features, you must run the MCP server alongside the main application.

**Terminal 1 (Run MCP Server):**
```bash
python run_mcp_server.py
```

**Terminal 2 (Run Main App):**
```bash
python main.py
```
The application will automatically build the necessary database tables on the first run.
Open your browser and navigate to `http://127.0.0.1:5000` to start chatting!

## 📂 Project Structure
* `app/models/`: Database schema definitions (User, Conversation, Message)
* `app/controllers/`: Route handlers for authentication and chat logic
* `app/services/`: Core logic for Vector searching, Web Fallback, and MCP client/server
* `app/views/`: HTML Templates and static assets
* `migrations/`: Database migration tracking
* `main.py`: Main application entry point
* `run_mcp_server.py`: Entry point for the Model Context Protocol engine
