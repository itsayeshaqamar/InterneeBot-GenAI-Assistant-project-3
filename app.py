import os
import re
import math
from collections import Counter
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from localized .env file
load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

# Global variables for RAG management
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_base.txt')
kb_chunks = []
vector_store = None
retriever = None
rag_status = {
    "loaded": False,
    "total_chunks": 0,
    "method": "Initializing...",
    "error": None
}

# Simple TF-IDF / Keyword Cosine Similarity Fallback Engine for fast, 100% reliable retrieval
class SimpleTFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = set()
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def _build_index(self):
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        for tokens in tokenized_docs:
            self.vocab.update(tokens)
        
        self.vocab = list(self.vocab)
        vocab_idx = {word: idx for idx, word in enumerate(self.vocab)}

        for tokens in tokenized_docs:
            counts = Counter(tokens)
            vec = [0] * len(self.vocab)
            for word, count in counts.items():
                if word in vocab_idx:
                    vec[vocab_idx[word]] = count
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            self.doc_vectors.append([x / norm for x in vec])

    def get_relevant_documents(self, query, top_k=3):
        query_tokens = self._tokenize(query)
        query_counts = Counter(query_tokens)
        vocab_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        q_vec = [0] * len(self.vocab)
        for word, count in query_counts.items():
            if word in vocab_idx:
                q_vec[vocab_idx[word]] = count
        
        q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        q_vec = [x / q_norm for x in q_vec]

        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            dot_product = sum(qv * dv for qv, dv in zip(q_vec, doc_vec))
            scores.append((dot_product, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        relevant_docs = []
        for score, idx in scores[:top_k]:
            if score > 0.05: # Minimum relevance threshold
                relevant_docs.append(self.documents[idx])
        
        # If no score above threshold, return top 2 doc snippets anyway
        if not relevant_docs and self.documents:
            relevant_docs = [self.documents[scores[0][1]], self.documents[scores[1][1]]] if len(self.documents) > 1 else self.documents
            
        return relevant_docs

def load_and_index_knowledge_base():
    global kb_chunks, vector_store, retriever, rag_status
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        rag_status["error"] = f"Knowledge base file not found at {KNOWLEDGE_BASE_PATH}"
        return

    try:
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # Split knowledge base by sections or paragraphs
        raw_sections = full_text.split("--------------------------------------------------------------------------------")
        chunks = []
        for sec in raw_sections:
            sec_clean = sec.strip()
            if len(sec_clean) > 50:
                # If section is long, break into sub-paragraphs
                sub_parts = sec_clean.split("\n\n")
                curr_chunk = ""
                for part in sub_parts:
                    if len(curr_chunk) + len(part) < 700:
                        curr_chunk += "\n" + part if curr_chunk else part
                    else:
                        chunks.append(curr_chunk.strip())
                        curr_chunk = part
                if curr_chunk:
                    chunks.append(curr_chunk.strip())

        kb_chunks = chunks
        rag_status["total_chunks"] = len(chunks)

        # Attempt to initialize LangChain FAISS / HuggingFace embedding retriever
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            lc_docs = text_splitter.create_documents([full_text])
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(lc_docs, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            rag_status["method"] = "LangChain FAISS + HuggingFace Embeddings"
            rag_status["loaded"] = True
            print("Successfully loaded LangChain FAISS RAG Retriever.")
        except Exception as faiss_err:
            print(f"FAISS/HuggingFace notice ({faiss_err}). Falling back to internal TF-IDF Retriever Engine.")
            retriever = SimpleTFIDFRetriever(kb_chunks)
            rag_status["method"] = "Internal Lightweight TF-IDF Retrieval Engine"
            rag_status["loaded"] = True

    except Exception as e:
        rag_status["error"] = str(e)
        print(f"Error loading Knowledge Base: {e}")

# Initialize Knowledge Base RAG index on startup
load_and_index_knowledge_base()

def get_context_for_query(query):
    """Retrieve top relevant context snippets from Knowledge Base"""
    if not retriever:
        return ""
    
    try:
        if hasattr(retriever, 'get_relevant_documents'):
            docs = retriever.get_relevant_documents(query)
            if docs:
                if isinstance(docs[0], str):
                    return "\n\n---\n\n".join(docs)
                else:
                    return "\n\n---\n\n".join([d.page_content for d in docs])
        elif hasattr(retriever, 'invoke'):
            docs = retriever.invoke(query)
            return "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as err:
        print(f"Error in context retrieval: {err}")
    
    # Fallback to simple keyword match
    fallback = SimpleTFIDFRetriever(kb_chunks)
    docs = fallback.get_relevant_documents(query, top_k=3)
    return "\n\n---\n\n".join(docs)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    groq_key = os.getenv("GROQ_API_KEY", "")
    key_configured = bool(groq_key and not groq_key.startswith("gsk_your_groq"))
    return jsonify({
        "status": "online",
        "model": "llama-3.1-8b-instant",
        "groq_key_configured": key_configured,
        "rag_status": rag_status
    })

@app.route('/api/quick-topics', methods=['GET'])
def get_quick_topics():
    topics = [
        {
            "id": "submissions",
            "title": "Task Submission Rules",
            "subtitle": "GitHub, LinkedIn post tagging & LMS upload guidelines",
            "icon": "code",
            "prompt": "How do I submit my internship tasks correctly? What are the GitHub and LinkedIn rules?"
        },
        {
            "id": "certificates",
            "title": "Certificates & LOR",
            "subtitle": "Completion certificate criteria, LOR & QR verification",
            "icon": "award",
            "prompt": "How can I qualify for an Internship Completion Certificate and Letter of Recommendation (LOR)?"
        },
        {
            "id": "deadlines",
            "title": "Deadline Extensions",
            "subtitle": "Requesting leaves, medical grace period & LMS unlocks",
            "icon": "clock",
            "prompt": "What is the policy for requesting a deadline extension or medical leave?"
        },
        {
            "id": "stipends",
            "title": "Stipends & Bounties",
            "subtitle": "Paid opportunity selection & performance badges",
            "icon": "zap",
            "prompt": "Are the internships paid? How are top performers selected for stipends and client projects?"
        }
    ]
    return jsonify({"topics": topics})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Message content cannot be empty."}), 400

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key or groq_api_key.startswith("gsk_your_groq"):
        return jsonify({
            "error": "GROQ_API_KEY is not configured in your .env file. Please add your Groq API key (starts with gsk_) to the .env file in the project folder.",
            "response": "⚠️ **Groq API Key Missing**\n\nPlease add your valid `GROQ_API_KEY` inside the `.env` file in the root workspace folder to enable live AI responses."
        }), 400

    # Retrieve relevant RAG context from Knowledge Base
    context_text = get_context_for_query(user_message)

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.2,
            max_tokens=1024
        )

        system_prompt_content = f"""You are 'InterneeBot', the official GenAI Virtual Assistant and Mentor Support Bot for Internee.pk interns.
Your primary role is to assist interns with clear, accurate, professional, and encouraging answers about Internee.pk policies, submission rules, task deadlines, certificates, and LMS portal workflows.

INSTRUCTIONS:
1. ALWAYS prioritize and base your response on the Official Internee.pk Knowledge Base context provided below.
2. If the answer is found in the context, synthesize it into a clean, well-formatted response using Markdown (bold headings, bullet points, step-by-step lists).
3. If the context does not fully cover the query, provide helpful general tech guidance while politely advising the intern to reach out via LMS Support or Discord for official account assistance.
4. Maintain a warm, encouraging, professional tone. Use relevant emojis occasionally to make responses engaging.

OFFICIAL INTERNEE.PK KNOWLEDGE BASE CONTEXT:
{context_text if context_text else 'No specific context retrieved.'}
"""

        messages = [SystemMessage(content=system_prompt_content)]

        # Include last 4 turns of conversation history for natural multi-turn dialogue
        for h in history[-4:]:
            role = h.get('role', '')
            content = h.get('content', '')
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_message))

        # Invoke Groq API
        ai_response = llm.invoke(messages)
        response_text = ai_response.content

        return jsonify({
            "response": response_text,
            "model": "llama-3.1-8b-instant",
            "context_retrieved": bool(context_text)
        })

    except Exception as e:
        print(f"Error calling Groq API: {e}")
        err_msg = str(e)
        if "api_key" in err_msg.lower() or "unauthorized" in err_msg.lower():
            return jsonify({
                "error": "Invalid Groq API Key provided. Please check your .env file.",
                "response": "❌ **Groq Authentication Error**: The provided GROQ_API_KEY appears invalid or expired. Please check your `.env` file."
            }), 401
        
        return jsonify({
            "error": str(e),
            "response": f"⚠️ **Connection Issue**: Unable to connect to Groq API ({str(e)}). Please try again in a few seconds."
        }), 500

@app.route('/api/checklist', methods=['POST'])
def generate_checklist():
    data = request.json or {}
    domain = data.get('domain', 'Web Development')
    task_num = data.get('task_number', '1')

    checklist = {
        "domain": domain,
        "task_number": task_num,
        "steps": [
            {
                "category": "1. Local Development & Setup",
                "items": [
                    f"Create dedicated project folder `internee-pk-{domain.lower().replace(' ', '-')}-task-{task_num}`",
                    "Initialize Git repository (`git init`) and create `.gitignore`",
                    "Ensure no secret API keys, `.env` or heavy vendor folders are tracked"
                ]
            },
            {
                "category": "2. Documentation & Quality",
                "items": [
                    "Create a comprehensive `README.md` with Project Title, Features & Setup commands",
                    "Include screenshots or a demo GIF in the README",
                    "Verify all links and functions work smoothly without errors"
                ]
            },
            {
                "category": "3. GitHub Push & Visibility",
                "items": [
                    "Push code to GitHub (`git push origin main`)",
                    "Ensure Repository Visibility is set to PUBLIC",
                    "Test opening repository link in Incognito browser window"
                ]
            },
            {
                "category": "4. LinkedIn Showcase (Mandatory)",
                "items": [
                    "Record a 30-60 second video demo or attach high-res screenshots",
                    "Mention @Internee.pk in the post body",
                    f"Add hashtags: `#InterneePK #VirtualInternship #{domain.replace(' ', '')} #PakistanTech`",
                    "Publish post and copy the public post URL"
                ]
            },
            {
                "category": "5. LMS Portal Upload",
                "items": [
                    "Log into https://lms.internee.pk",
                    "Submit GitHub Repository Link & LinkedIn Post Link",
                    "Include live hosted link (Vercel/Netlify/Render) if web application",
                    "Confirm submission before official weekly deadline"
                ]
            }
        ]
    }
    return jsonify(checklist)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"\n[SERVER ONLINE] InterneeBot GenAI Server running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
