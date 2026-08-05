# 📚 InterneeBot - Technical Architecture & Documentation

This document provides a deep dive into the technical architecture, component breakdown, RAG (Retrieval-Augmented Generation) retrieval workflow, data flow sequence, and security considerations of **InterneeBot**.

---

## 🏗️ System Architecture Overview

InterneeBot follows a modular **Retrieval-Augmented Generation (RAG)** architecture using a Flask web backend, LangChain orchestration, Groq LPU inference, and a modern Tailwind CSS glassmorphic frontend.

```mermaid
graph TD
    User([Intern User]) -->|1. Types query / Uses Voice Dictation| UI[Tailwind CSS Glassmorphism Frontend]
    UI -->|2. POST /api/chat {message, history}| Flask[Flask Backend Web Server]
    
    subgraph Backend Pipeline app.py
        Flask -->|3. Retrieve relevant sections| RAG[RAG Retrieval Engine]
        KB[(data/knowledge_base.txt)] -->|Index 24 Sections| RAG
        RAG -->|4. Top Matching Context| Prompt[System Prompt Generator]
        Prompt -->|5. Context + Memory + User Query| LC[LangChain ChatGroq Interface]
    end
    
    LC -->|6. API Request llama-3.1-8b-instant| Groq[Groq API LPU Cluster]
    Groq -->|7. Generated Response| LC
    LC -->|8. Formatted Response| Flask
    Flask -->|9. JSON Response {response, model}| UI
    UI -->|10. Markdown Render & TTS Playback| User
```

---

## 🧠 Retrieval-Augmented Generation (RAG) Architecture

To eliminate LLM hallucinations and guarantee that all answers strictly adhere to official Internee.pk guidelines, InterneeBot implements a localized RAG context injection engine.

### 1. Document Loading & Sectioning
- Reads `data/knowledge_base.txt` upon application boot.
- Splits knowledge content into structured, logical chunks (sections, paragraphs, FAQs).
- Assigns section identifiers and metadata to each chunk.

### 2. Dual Retrieval Strategy
The system features a dual-engine retriever setup for maximum performance and zero dependency overhead:
1. **LangChain FAISS & HuggingFace Embeddings Interface:** Supports dense vector similarity search using `all-MiniLM-L6-v2`.
2. **Lightweight Keyword TF-IDF Retriever Engine:** Built-in zero-dependency TF-IDF cosine similarity search engine that tokenizes documents, calculates document frequency weights, and computes cosine distance against user queries.
   - **Benefit:** Guarantees 100% server uptime and instantaneous startup time without requiring heavy 100MB+ PyTorch binary downloads on lightweight cloud instances.

### 3. Contextual Prompt Construction
When a query arrives, the retriever extracts the top-K most relevant knowledge base snippets ($k=3$) and formats them into a strict System Prompt:

```text
You are 'InterneeBot', the official GenAI Virtual Assistant and Mentor Support Bot for Internee.pk interns.
Your primary role is to assist interns with clear, accurate, professional, and encouraging answers about Internee.pk policies, submission rules, task deadlines, certificates, and LMS portal workflows.

INSTRUCTIONS:
1. ALWAYS prioritize and base your response on the Official Internee.pk Knowledge Base context provided below.
2. Synthesize answers into a clean, well-formatted response using Markdown (bold headings, bullet points, step-by-step lists).
3. Maintain a warm, encouraging, professional tone.

OFFICIAL INTERNEE.PK KNOWLEDGE BASE CONTEXT:
[Retrieved Knowledge Base Snippets]
```

---

## 🛠️ Backend API Endpoints Specification

### `POST /api/chat`
- **Description:** Primary conversational endpoint handling RAG retrieval and Groq LLM inference.
- **Request Body:**
  ```json
  {
    "message": "What are the rules for submitting my task on LMS and LinkedIn?",
    "history": [
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
  }
  ```
- **Response Body:**
  ```json
  {
    "response": "**Task Submission Guidelines**\n\nTo submit your weekly task...",
    "model": "llama-3.1-8b-instant",
    "context_retrieved": true
  }
  ```

### `GET /api/status`
- **Description:** System health diagnostic check.
- **Response Body:**
  ```json
  {
    "status": "online",
    "model": "llama-3.1-8b-instant",
    "groq_key_configured": true,
    "rag_status": {
      "loaded": true,
      "total_chunks": 24,
      "method": "Internal Lightweight TF-IDF Retrieval Engine",
      "error": null
    }
  }
  ```

### `POST /api/checklist`
- **Description:** Domain-specific task submission checklist generator.
- **Request Body:** `{"domain": "Web Development", "task_number": "1"}`
- **Response Body:**
  ```json
  {
    "domain": "Web Development",
    "task_number": "1",
    "steps": [
      {
        "category": "1. Local Development & Setup",
        "items": [
          "Create dedicated project folder internee-pk-web-development-task-1",
          "Initialize Git repository and add .gitignore"
        ]
      }
    ]
  }
  ```

---

## 🎨 Frontend Architecture & UX Engineering

The frontend is implemented in `templates/index.html` as a Single-Page Application (SPA) using vanilla JavaScript and Tailwind CSS.

### Key Components:
1. **Sidebar Navigation Drawer (`#sidebar`):**
   - New Chat command (`⌘N`).
   - Quick topic search filter.
   - 🚀 Direct Task Portal launcher (`openLmsPortal()`).
   - Interactive Tool Modals: Task Checklist Tool, Knowledge Index Browser, Support Ticket Formatter.
2. **Chat Container & Empty Hero State:**
   - 3D Floating Orb element (`.orb-gradient`) with subtle floating and glowing keyframe animations.
   - Quick suggestion prompt cards.
   - Auto-scrolling chat history stream (`#chat-messages`).
3. **Interactive Messaging System:**
   - Markdown parsing using `Marked.js`.
   - Syntax highlighting using `Highlight.js`.
   - Text-to-Speech (read aloud via Web Speech API `speechSynthesis`).
   - Speech-to-Text (microphone dictation via Web Speech API `SpeechRecognition`).
   - Code block 1-click copy buttons.

---

## 🔐 Security & Best Practices

1. **Environment Isolation:**
   - API Keys are strictly loaded from `.env` via `python-dotenv`.
   - `.gitignore` explicitly excludes `.env`, `venv/`, cache folders, and log files to prevent secret leakage.
2. **Input Sanitization:**
   - User inputs are escaped prior to rendering in raw HTML elements to prevent Cross-Site Scripting (XSS).
   - Input length validation prevents buffer overflow or API spam.
3. **CORS & Rate Limits:**
   - Flask-CORS enabled for controlled API access.

---

## 🚀 Production Deployment Guidelines

To deploy InterneeBot to production environments like **Render**, **Railway**, or **AWS EC2**:

1. **Use Gunicorn WSGI Server:**
   ```bash
   gunicorn --bind 0.0.0.0:5000 app:app
   ```
2. **Set Environment Variables on Hosting Platform:**
   - Configure `GROQ_API_KEY` in environment secret settings.
3. **Configure SSL / HTTPS:**
   - Ensure reverse proxy (Nginx or Cloudflare) forces HTTPS traffic.
