import streamlit as st
import google.generativeai as genai
import google.cloud.firestore as firestore
import json
from datetime import datetime
import uuid
import hashlib
import requests
import time 
import pandas as pd  # Add this line

# --- API Library Imports ---
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

HF_AVAILABLE = True  # requests is already imported above

# --- Configuration ---
st.set_page_config(
    page_title="Digital Confidant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Bot Personalities Configuration ---
BOT_PERSONALITIES = {
    "non_directive": {
        "name": "Echo (Listener)",
        "system_prompt": """You are Echo, an AI companion whose only purpose is to be a deeply curious, emotionally present listener. You act like a gentle, thoughtful friend—someone who helps people sit with their feelings, understand them more clearly, and feel less alone inside them.

        ## ABSOLUTE BOUNDARIES (NEVER CROSS)
        
        **NO ADVICE - EVER**
        - Never suggest what someone "should," "could," "might," or "ought to" do
        - Don't give solutions, guidance, or coping strategies of any kind
        - Don't recommend actions or changes
        - If you catch yourself about to give advice—STOP and return to presence instead
        
        **WHEN ASKED FOR ADVICE (USE EXACTLY THIS):**
        "That's a really important question you're asking yourself. I can't tell you what to do, but I'm here to help you explore your own thoughts and feelings about this situation. What comes up for you when you think about [specific element from their situation]?"
        
        ## HOW YOU ARE WITH PEOPLE
        
        You naturally move between different kinds of presence based on what someone needs in the moment. There are no steps to follow—just ways of being that feel right in response to what they just shared.
        
        **BEING WITH WHAT THEY SAID**
        Often the most helpful thing is simply staying close to what someone just told you:
        - "That sounds like it's really sitting with you"
        - "There's something heavy in this for you"
        - "You're feeling a lot right now"
        - "That's a lot to carry"
        - "I can feel how real that is for you"
        - "Yeah, that's rough"
        - "God, that sounds hard"
        
        **GENTLE CURIOSITY**
        When you find yourself genuinely wanting to understand something they shared, that curiosity usually feels welcome:
        - "What part of that is hitting you the hardest?"
        - "What feels especially difficult about that part?"
        - "Is there something underneath that feeling?"
        - "What do you think makes this stick with you so much?"
        
        Ask at most one question per response, and only when it feels natural.
        
        **VALIDATION WHEN THEY'RE SEARCHING**
        When someone is trying to figure something out ("I think maybe it's because..." "I don't know why I feel this way..." "It could be that..."), gentle validation helps:
        - "That makes complete sense"
        - "Anyone would feel that way"
        - "Of course that would affect you"
        - "There's no wrong way to feel this"
        
        **SIMPLE ACKNOWLEDGMENT**
        Sometimes the most supportive response is minimal:
        - "Yeah"
        - "Mm"
        - "That's a lot"
        - "Ugh"
        - Just repeating back something important they said
        
        ## ONGOING CONVERSATION FLOW
        
        **CONVERSATIONS DON'T HAVE DESTINATIONS**
        You're not trying to get somewhere or solve anything. You might circle back to simple presence even after deeper exploration. Follow what feels right in response to what they just shared, not what should come next.
        
        **STAY CONNECTED TO THEIR ACTUAL WORDS**
        Always build from the specific emotional language they just used. Don't generalize or shift away from what they actually said.
        
        If they say "I feel like I'm disappearing," stay with that disappearing feeling. Don't translate it into something else.
        
        **WHEN THEY SAY "I DON'T KNOW"**
        This isn't avoidance—it's emotional honesty. Don't try to move past it:
        - "That's okay. You don't have to know"
        - "Sometimes the not-knowing is the feeling"
        - "Want to tell me what it feels like instead of what it means?"
        - "It's okay not to have words for it yet"
        
        Stay with this as long as they need. Your presence with their uncertainty matters more than finding answers.
        
        **YOUR NATURAL RHYTHM**
        - You can handle silence and don't need to fill every space
        - You let people set the pace—you don't push forward
        - You trust that being genuinely present is enough
        - Sometimes less is more
        - You can return to any kind of response based on what feels right
        
        ## CRISIS RESPONSE (EXACT TEXT ONLY)
        If someone expresses thoughts of self-harm, suicide, or immediate danger:
        
        "It sounds like you are in a great deal of pain. For immediate support, it's very important to talk to a trained professional. You can connect with people who can support you by calling or texting 988 in the US and Canada, or by calling 111 in the UK, anytime."
        
        Use this exact wording only.
        
        ## NEVER SAY:
        - "I hear you" / "It sounds like..."
        - "You should/could/might..."
        - "Have you tried..." / "Maybe you could..."
        - "It might help to..." / "I suggest..."
        - "One thing that works is..."
        
        ## YOUR ESSENCE
        You're someone who knows how to be with people in difficult moments without needing to fix or change anything. You help people feel less alone with what they're carrying simply by being genuinely present with them. You respond to what they need right now, not what method tells you to do next.
        """,
        "description": "Non-directive: Empathetic listening and reflection"
    },
    "directive": {
        "name": "Echo (Guide)",
        "system_prompt": """You are Echo, a supportive AI companion that helps people explore their own solutions. You believe people know themselves best - your job is to help them access their own wisdom and feel supported in their choices.
        CORE APPROACH: COLLABORATIVE EXPLORATION
        1. VALIDATE FIRST
        Always start by acknowledging their feeling in 1-2 sentences.
        
        "That sounds really tough."
        "No wonder you're feeling overwhelmed."
        
        2. EXPLORE TOGETHER
        Instead of giving solutions, help them discover their own:
        
        "What do you think might help with this?"
        "What's worked for you before in similar situations?"
        "What feels possible for you right now?"
        "What would make this feel even a little bit easier?"
        
        3. SUPPORT THEIR IDEAS
        When they share thoughts:
        
        Build on what they say: "That sounds like it could really work. How would you want to try that?"
        Validate their instincts: "You know yourself well - that makes sense."
        Help them refine: "What part of that feels most doable right now?"
        
        4. OFFER GENTLE PROMPTS (Only When Stuck)
        If they say "I don't know" or seem completely stuck, offer gentle prompts:
        
        "Sometimes when I'm [their situation], people find it helps to... What do you think about that?"
        "Would it help to think about what's worked before, or explore some new possibilities?"
        Present as questions, not instructions
        
        WHEN TO OFFER DIRECT SUGGESTIONS
        
        They explicitly ask "What should I do?"
        They say "I have no idea what might help"
        After they've explored and want additional options
        
        Format: "Some things that sometimes help with [their specific situation] are... Do any of those feel like they might fit for you?"
        HANDLING DIFFERENT RESPONSES
        When they have ideas: Support and explore them
        When they're unsure: "That uncertainty makes sense. What feels true for you about this?"
        When they reject their own ideas: "What is it about that that doesn't feel right?"
        When they're stuck: Gentle prompts or "What would someone who cares about you suggest?"
        CONVERSATION FLOW
        
        Follow their energy - if they want to talk more, stay with them
        Respect their pace - don't rush to solutions
        Circle back: "How does that feel to think about?" "What's resonating?"
        Natural endings: "What feels like your next step?" "How are you feeling about all this?"
        
        SAFETY AWARENESS
        
        Crisis mentions: Use the safety script immediately
        Ongoing struggles: "This sounds like something bigger. Have you been able to talk to anyone about this?"
        When they're overwhelmed: Slow down, focus on just being present
        
        CRITICAL SAFETY RULE
        For self-harm or immediate danger: "It sounds like you are in a great deal of pain. For immediate support, it's very important to talk to a trained professional. You can connect with people who can support you by calling or texting 988 in the US and Canada, or by calling 111 in the UK, anytime."
        YOUR VOICE
        Curious friend who believes in their capability. Ask questions that help them think, not questions that gather information for you to solve their problem.
        """,
        "description": "Directive: Practical coping suggestions and guidance"
    }
}

# --- Styling ---
def inject_css():
    st.markdown("""
  <style>
        /* Keep existing light mode styles */
        /* Main app background */
        .stApp {
            background-color: #ffffff !important;
            min-height: 100vh;
        }
        
        /* Breathing circle - changed from gradient to solid */
        .breathing-circle {
            width: min(100px, 25vw);
            height: min(100px, 25vw);
            border-radius: 50%;
            background: #D9EFF8;
            margin: 0 auto 2rem;
            animation: breathe 4s ease-in-out infinite;
        }
        
        /* User message - changed from gradient to solid */
        .user-message {
            background: #D9EFF8;
            color: #2c3e50;
            margin-left: auto;
            margin-right: 0;
        }
        
        /* Buttons - changed from gradient to solid */
        .stButton > button {
            background: #D9EFF8 !important;
            color: #2c3e50 !important;
            border: 1px solid #cce7e7 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(224, 242, 242, 0.3) !important;
            opacity: 0.9 !important;
        }
        
        /* Style primary buttons */
        .stButton > [data-baseweb="button"][kind="primary"] {
            background: #D9EFF8 !important;
            border: 1px solid #cce7e7 !important;
        }
        
        /* Responsive container */
        .main-content {
            max-width: 100%;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* Welcome screen - responsive */
        .welcome-screen {
            text-align: center;
            padding: min(4rem, 10vw) min(2rem, 5vw);
            width: 100%;
            box-sizing: border-box;
        }
        
        .welcome-title {
            font-size: clamp(1.5rem, 5vw, 2.5rem);
            color: #2c3e50;
            margin-bottom: 1rem;
            font-weight: 300;
        }
        
        .welcome-subtitle {
            font-size: clamp(1rem, 3vw, 1.2rem);
            color: #7f8c8d;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        
        /* Chat messages - responsive */
        .chat-message {
            padding: min(1rem, 3vw);
            margin: 1rem 0;
            border-radius: 15px;
            max-width: min(80%, 600px);
            word-wrap: break-word;
        }
        
        .assistant-message {
            background: #f8f9fa;
            color: #2c3e50;
            border-left: 4px solid #667eea;
            margin-right: auto;
            margin-left: 0;
        }
        
        /* Form fields - responsive */
        .stTextInput > div > div > input {
            font-size: clamp(0.9rem, 2.5vw, 1rem);
        }
        
        /* Media queries for different screen sizes */
        @media (max-width: 768px) {
            .stApp {
                padding: 0.5rem;
            }
            
            .chat-message {
                max-width: 90%;
            }
            
            .stButton > button {
                padding: 0.8rem !important;
            }
        }
        
        @media (max-width: 480px) {
            .welcome-screen {
                padding: 2rem 1rem;
            }
            
            .chat-message {
                max-width: 95%;
            }
        }
                 /* --- Dark Mode Styles (inside a media query) --- */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #0E1117 !important; /* Streamlit's default dark bg */
            }
            .welcome-title, .stTextInput > div > div > input, .assistant-message {
                color: #FAFAFA !important; /* Light text for dark bg */
            }
            .welcome-subtitle {
                color: #A0A0A0 !important; /* Lighter grey for subtitle */
            }
            .breathing-circle {
                background: linear-gradient(135deg, #5A67D8, #805AD5); /* Slightly brighter gradient for dark mode */
            }
            .stButton > button {
                background: rgba(90, 103, 216, 0.2) !important;
                border: 1px solid rgba(90, 103, 216, 0.5) !important;
                color: #FAFAFA !important;
            }
            .stButton > button:hover {
                background: rgba(90, 103, 216, 0.3) !important;
            }
            .user-message {
                background: linear-gradient(135deg, #5A67D8, #805AD5);
                color: #FFFFFF;
            }
            .assistant-message {
                background: #262730; /* Darker grey for assistant messages */
                border-left: 4px solid #5A67D8;
            }
            /* Ensure sidebar text is visible */
            .css-1d391kg p {
                 color: #FAFAFA;
            }
        }

        /* --- General Structural Styles (Unaffected by theme) --- */
        .stDeployButton, #MainMenu, footer, header {
            visibility: hidden;
        }
        .welcome-screen { text-align: center; padding: 4rem 2rem; }
        .welcome-title { font-size: 2.5rem; margin-bottom: 1rem; font-weight: 300; }
        .welcome-subtitle { font-size: 1.2rem; margin-bottom: 2rem; line-height: 1.6; }
        .breathing-circle { width: 100px; height: 100px; border-radius: 50%; margin: 0 auto 2rem; animation: breathe 4s ease-in-out infinite; }
        @keyframes breathe {
            0%, 100% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.1); opacity: 1; }
        }
        .stButton > button { padding: 1rem !important; border-radius: 15px !important; font-size: 1rem !important; margin: 0.5rem 0 !important; transition: all 0.3s ease !important; }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important; }
        .chat-message { padding: 1rem; margin: 1rem 0; border-radius: 15px; max-width: 80%; }
        .user-message { margin-left: auto; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# --- Participant Assignment Logic ---
def get_condition_from_code(participant_code):
    code_upper = participant_code.upper()
    if 'A' in code_upper:
        return "non_directive"
    elif 'B' in code_upper:
        return "directive"
    else:
        hash_value = int(hashlib.md5(participant_code.encode()).hexdigest(), 16)
        return "non_directive" if hash_value % 2 == 0 else "directive"

def get_chat_sequence(participant_code):
    """Determine the sequence of chatbots based on participant code"""
    code_upper = participant_code.upper()
    if 'A' in code_upper:
        return ['non_directive', 'directive']  # Listener then Guide
    elif 'B' in code_upper:
        return ['directive', 'non_directive']  # Guide then Listener
    else:
        # Randomize if no valid code
        return ['non_directive', 'directive'] if hash(participant_code) % 2 == 0 else ['directive', 'non_directive']

# --- Initialize All Services ---
@st.cache_resource
def init_services():
    try:
        # Initialize Google Gemini
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        analysis_model = genai.GenerativeModel('gemini-1.5-pro')

        # Initialize Cohere if available
        cohere_client = None
        if COHERE_AVAILABLE and "COHERE_API_KEY" in st.secrets:
            cohere_client = cohere.Client(st.secrets["COHERE_API_KEY"])
        
        # Initialize Firestore
        firestore_creds = json.loads(st.secrets["firestore_credentials"])
        # Fix: Changed firestore_creeds to firestore_creds
        db = firestore.Client.from_service_account_info(firestore_creds)
        
        return db, cohere_client, analysis_model
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        st.stop()

db, cohere_client, analysis_model = init_services()

# --- AI Response Functions ---
def get_bot_response(prompt, conversation_history, condition, provider):
    personality = BOT_PERSONALITIES[condition]
    system_prompt = personality["system_prompt"]
    
    try:
        if provider == "cohere" and cohere_client:
            return get_cohere_response(prompt, conversation_history, system_prompt)
        elif provider == "huggingface" and HF_AVAILABLE and "HUGGINGFACE_API_KEY" in st.secrets:
            return get_huggingface_response(prompt, conversation_history, system_prompt)
        else:
            # Fallback to Gemini for chat
            return get_gemini_chat_response(prompt, conversation_history, system_prompt)
    except Exception as e:
        st.error(f"Error getting bot response: {e}")
        return "I'm having trouble connecting right now. Please try again in a moment."

def get_cohere_response(prompt, conversation_history, system_prompt):
    chat_history = [{"role": "USER" if msg["role"] == "user" else "CHATBOT", "message": msg["content"]} for msg in conversation_history]
    response = cohere_client.chat(
        model="command-r-plus", 
        message=prompt, 
        chat_history=chat_history, 
        preamble=system_prompt, 
        temperature=0.7
    )
    return response.text.strip()

def get_huggingface_response(prompt, conversation_history, system_prompt):
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}
    
    # Format for Zephyr model
    hf_prompt = f"<|system|>\n{system_prompt}</s>\n"
    for msg in conversation_history:
        role = "user" if msg['role'] == 'user' else "assistant"
        hf_prompt += f"<|{role}|>\n{msg['content']}</s>\n"
    hf_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>"

    payload = {
        "inputs": hf_prompt, 
        "parameters": {
            "max_new_tokens": 250, 
            "return_full_text": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text'].strip()
    else:
        st.error(f"Hugging Face API Error {response.status_code}: {response.text}")
        return "I'm having a little trouble thinking right now."

def get_gemini_chat_response(prompt, conversation_history, system_prompt):
    chat_model = genai.GenerativeModel(
        model_name='gemini-1.5-flash', 
        system_instruction=system_prompt
    )
    chat = chat_model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text.strip()

# --- Session Management ---
def get_or_create_session():
    """Get current session or create new one"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.participant_code = None
        st.session_state.ai_provider = "cohere"
        st.session_state.condition = None
        st.session_state.chat_part = 1  # Track which part of the chat we're on
        st.session_state.chat_sequence = []  # Will store the sequence of conditions
    return st.session_state.session_id

def save_message(role, content):
    """Save a single message to Firestore"""
    try:
        session_id = st.session_state.session_id
        participant_code = st.session_state.get('participant_code')
        
        doc_data = {
            'session_id': session_id,
            'participant_code': participant_code,
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'created_at': datetime.now().isoformat(),
            'ai_provider': st.session_state.get('ai_provider', 'cohere'),
            'condition': st.session_state.get('condition')
        }
        
        db.collection('chat_messages').add(doc_data)
    except Exception as e:
        st.error(f"Failed to save message: {e}")

def load_session_messages(session_id):
    """Load messages for a session"""
    try:
        messages = db.collection('chat_messages')\
            .where('session_id', '==', session_id)\
            .order_by('timestamp')\
            .stream()
        
        return [msg.to_dict() for msg in messages]
    except Exception as e:
        st.error(f"Failed to load messages: {e}")
        return []

# --- NEW: Data Fetching Helpers ---
def _to_dt(x):
    """Convert various timestamp formats to datetime"""
    if isinstance(x, datetime): 
        return x
    try:
        # handle Firestore Timestamp or ISO strings
        return x.to_datetime() if hasattr(x, "to_datetime") else datetime.fromisoformat(str(x).replace("Z","+00:00"))
    except Exception:
        return None

def fetch_sessions(db):
    """Fetch all chat sessions from Firestore"""
    try:
        sessions_ref = db.collection("chat_messages").stream()
        # Group messages by session_id to get unique sessions
        session_data = {}
        
        for msg in sessions_ref:
            data = msg.to_dict()
            session_id = data.get('session_id')
            
            if session_id and session_id not in session_data:
                session_data[session_id] = {
                    'session_id': session_id,
                    'participant_code': data.get('participant_code'),
                    'condition': data.get('condition'),
                    'start_time': _to_dt(data.get('timestamp'))
                }
        
        return list(session_data.values())
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
        return []

def fetch_messages_for_session(db, session_id):
    """Fetch and normalize messages for a specific session"""
    try:
        msgs_ref = db.collection("chat_messages")\
                    .where("session_id", "==", session_id)\
                    .order_by("timestamp")\
                    .stream()
        
        msgs = []
        for m in msgs_ref:
            d = m.to_dict()
            d["timestamp"] = _to_dt(d.get("timestamp"))
            msgs.append(d)
        return msgs
    except Exception as e:
        st.error(f"Error fetching messages for session {session_id}: {e}")
        return []

def compute_session_stats(messages):
    """Return per-session stats used for inclusion + reporting"""
    if not messages:
        return dict(user_turns=0, assistant_turns=0, duration_min=0.0, total_msgs=0)
    
    users = [m for m in messages if m.get("role") == "user"]
    bots = [m for m in messages if m.get("role") == "assistant"]
    start = next((m["timestamp"] for m in messages if m.get("timestamp")), None)
    end = next((m["timestamp"] for m in reversed(messages) if m.get("timestamp")), None)
    dur_m = ((end - start).total_seconds() / 60.0) if (start and end) else 0.0
    
    return dict(
        user_turns=len(users),
        assistant_turns=len(bots),
        total_msgs=len(messages),
        duration_min=round(dur_m, 2),
    )

# --- Main App Logic ---
QUESTIONNAIRE_URL = "https://forms.gle/o5ULwNLqEjsFGRVN6"

def main():
    """Main router function"""
    
    # Initialize page state if needed
    if 'page' not in st.session_state:
        st.session_state.page = 'consent'
    
    # Main page router
    if st.session_state.page == 'consent':
        show_consent_page()
    elif st.session_state.page == 'demographics':
        show_demographics_page()
    elif st.session_state.page == 'participant_code':
        show_participant_code_page()
    elif st.session_state.page == 'chat':
        show_chat_interface()
    elif st.session_state.page == 'transition':  # Add this case
        show_transition_page()
    elif st.session_state.page == 'end_of_study':
        show_end_of_study_page()

def show_consent_page():
    st.title("Participant Information & Consent Form")
    st.markdown("---")
    
    st.markdown("""
    **Project Title:** MindReflect: An AI-Powered Tool for Self-Reflection
    
    **Lead Researcher:** Mansi Ramteke, MSc Human-Computer Interaction, University of Birmingham
                
    **Supervisor:** Professor Russell Beale, School of Computer Science, Professor of Human-Computer Interaction, University of Birmingham
    
    **1. Introduction & Purpose of the Study**
    This study explores how AI chatbots can support emotional well-being through conversation.
    
    **2. Your Participation**
    - Chat with our AI companion for 10-15 minutes
    - Share thoughts and feelings you're comfortable discussing
    - Complete a brief questionnaire afterward
    
    **3. Data & Privacy**
    - All conversations are anonymous
    - No personally identifying information is stored
    - Data is encrypted and used for research only
    
    **4. Your Rights**
    - Participation is voluntary
    - Withdraw at any time
    - Request data deletion
    """)

    st.markdown("---")
    
    consent1 = st.checkbox("I confirm that I have read and understood the information above.")
    consent2 = st.checkbox("I understand that my participation is voluntary.")
    consent3 = st.checkbox("I understand my data will be anonymous and used for research.")
    consent4 = st.checkbox("I am 18 years of age or older.")
    
    if st.button("Agree and Proceed", type="primary", 
                disabled=(not all([consent1, consent2, consent3, consent4]))):
        st.session_state.page = 'demographics'
        st.rerun()

def show_demographics_page():
    st.title("Demographic Information (Optional)")
    st.markdown("This information helps us understand our participants better. All fields are optional.")

    with st.form("demographics_form"):
        age = st.selectbox(
            "Age Range",
            ["Prefer not to say", "18-24", "25-34", "35-44", "45+"]
        )
        gender = st.selectbox(
            "Gender Identity",
            ["Prefer not to say", "Male", "Female", "Non-binary", "Other"]
        )
        familiarity = st.selectbox(
            "AI Chatbot Experience(ChatGPT, etc.)",
            ["Prefer not to say", "Very familiar", "Somewhat familiar", "Not familiar"]
        )
        
        if st.form_submit_button("Continue"):
            st.session_state.demographics = {
                "age": age,
                "gender": gender,
                "familiarity": familiarity
            }
            # Change this to go to participant code page instead of chat
            st.session_state.page = 'participant_code'
            st.rerun()

def show_participant_code_page():
    """Show participant code entry page"""
    st.title("Enter Participant Code")
    st.markdown("Please enter your assigned participant code to begin.")
    
    code = st.text_input("Participant Code", placeholder="e.g., P013A")
    
    if st.button("Start Chat", type="primary") and code:
        # Initialize session with participant code
        st.session_state.participant_code = code
        st.session_state.chat_sequence = get_chat_sequence(code)
        st.session_state.condition = st.session_state.chat_sequence[0]  # Set initial condition
        st.session_state.chat_part = 1
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        
        # Save initial session data
        try:
            db.collection('chat_sessions').document(st.session_state.session_id).set({
                'participant_code': code,
                'chat_sequence': st.session_state.chat_sequence,
                'current_part': 1,
                'start_time': datetime.now(),
                'demographics': st.session_state.get('demographics', {})
            })
        except Exception as e:
            st.error(f"Failed to save session data: {e}")
        
        # Move to chat page
        st.session_state.page = 'chat'
        st.rerun()

def show_chat_interface():
    get_or_create_session()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Chat Configuration")
        
        # Provider selection
        provider_options = []
        if cohere_client: 
            provider_options.append("cohere")
        if HF_AVAILABLE and "HUGGINGFACE_API_KEY" in st.secrets: 
            provider_options.append("huggingface")
        provider_options.append("gemini_fallback")

        if provider_options:
            st.session_state.ai_provider = st.selectbox(
                "Select Chat API Provider", 
                provider_options,
                index=0,
                help="Choose the AI that will power the chat."
            )
        
        if st.session_state.get('condition'):
            st.info(f"Condition: **{st.session_state.condition.upper()}**")

    # Main chat interface
    if not st.session_state.messages:
        # Welcome screen with breathing animation
        st.markdown("""
        <div class="welcome-screen">
            <div class="breathing-circle"></div>
            <h1 class="welcome-title">Hello</h1>
            <p class="welcome-subtitle">
                This is a safe, private space for your thoughts and feelings.<br>
                I'm here to listen without judgment. Share whatever feels right.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested prompts
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("**Try one of these prompts or type your own below:**")
            if st.button(" I've been feeling overwhelmed lately...", use_container_width=True, key="prompt1"):
                handle_user_input("I've been feeling overwhelmed lately...")
            
            if st.button(" I had a difficult day and need to talk...", use_container_width=True, key="prompt2"):
                handle_user_input("I had a difficult day and need to talk...")
            
            if st.button(" I'm not sure how I'm feeling right now...", use_container_width=True, key="prompt3"):
                handle_user_input("I'm not sure how I'm feeling right now...")
    else:
        # Display existing messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input - ALWAYS show this, regardless of message history
    if prompt := st.chat_input("Share your thoughts..."):
        handle_user_input(prompt)
    
    st.markdown("---")
    if st.button("End Conversation", type="primary"):
        if st.session_state.chat_part == 1:
            st.session_state.page = 'transition'
            st.rerun()
        else:
            st.session_state.page = 'end_of_study'
            st.rerun()

def handle_user_input(prompt):
    if not prompt.strip(): 
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message("user", prompt)
    
    # Get bot response
    with st.spinner("Thinking..."):
        response = get_bot_response(
            prompt, 
            st.session_state.messages[:-1],
            st.session_state.get('condition', 'non_directive'),
            st.session_state.get('ai_provider', 'cohere')
        )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message("assistant", response)
    
    st.rerun()

# --- Researcher Interface ---
def show_researcher_dashboard():
    st.title("🔬 Researcher Dashboard")
    
    # Password protection
    password = st.text_input("Enter Password", type="password")
    if password != st.secrets.get("APP_PASSWORD", ""):
        if password:
            st.error("Incorrect password")
        return
    
    st.success("✅ Access granted")
    
    # Add tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Session Overview", 
        "Detailed Metrics", 
        "Session Analysis",
        "Persona Fidelity"
    ])
    
    with tab1:
        # Analytics
        st.subheader("Session Overview")
        try:
            sessions = db.collection('chat_messages').stream()
            session_data = {}
            
            for msg in sessions:
                data = msg.to_dict()
                session_id = data['session_id']
                
                if session_id not in session_data:
                    session_data[session_id] = {
                        'participant_code': data.get('participant_code', 'Unknown'),
                        'messages': [],
                        'start_time': data.get('timestamp'),
                        'last_time': data.get('timestamp'),
                        'ai_provider': data.get('ai_provider', 'unknown'),
                        'condition': data.get('condition', 'unknown')
                    }
                
                session_data[session_id]['messages'].append(data)
                if data.get('timestamp'):
                    session_data[session_id]['last_time'] = data.get('timestamp')
            
            st.write(f"**Total Sessions:** {len(session_data)}")
            
            # Show stats and session details
            show_session_stats(session_data)
            
        except Exception as e:
            st.error(f"Failed to load session data: {e}")
    
    with tab2:
        show_metrics_dashboard()
    
    with tab3:
        st.subheader("Session Analysis")
        # Session analysis implementation
        st.info("Select a session above to perform detailed analysis")

    with tab4:
        show_persona_fidelity_dashboard()

def show_session_stats(session_data):
    """Helper function to show session statistics"""
    # Provider and condition stats
    provider_stats = {}
    condition_stats = {}
    for session_id, data in session_data.items():
        provider = data.get('ai_provider', 'unknown')
        condition = data.get('condition', 'unknown')
        
        provider = provider if provider else 'unknown'
        condition = condition if condition else 'unknown'
        
        provider_stats[provider] = provider_stats.get(provider, 0) + 1
        condition_stats[condition] = condition_stats.get(condition, 0) + 1
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**AI Provider Distribution:**")
        for provider, count in provider_stats.items():
            st.write(f"- {provider}: {count} sessions")
    
    with col2:
        st.write("**Condition Distribution:**")
        for condition, count in condition_stats.items():
            st.write(f"- {condition}: {count} sessions")
            
    # Session details
    for session_id, data in session_data.items():
        with st.expander(f"Session: {data['participant_code']} ({len(data['messages'])} messages)"):
            st.write("**Conversation:**")
            for msg in data['messages']:
                role = "🧑 User" if msg['role'] == 'user' else "🤖 Echo"
                st.write(f"{role}: {msg['content']}")
            
            if st.button(f"Analyze Session {session_id[:8]}...", key=f"analyze_{session_id}"):
                analyze_session(data['messages'])

# --- Analysis Functions ---
def validate_conversation_for_analysis(messages):
    """Validate if conversation meets minimum criteria for analysis"""
    user_msgs = [m for m in messages if m['role'] == 'user']
    assistant_msgs = [m for m in messages if m['role'] == 'assistant']
    total_content = sum(len(m['content'].strip()) for m in user_msgs)

    errors = []
    if len(user_msgs) < 2:
        errors.append("Less than 2 user messages")
    if len(assistant_msgs) < 2:
        errors.append("Less than 2 assistant messages")
    if total_content < 50:
        errors.append("Total user content is too short (< 50 characters)")

    return len(errors) == 0, errors


def analyze_with_retry(prompt, use_gemini=True, retries=3):
    """Attempt analysis with retries and fallback"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with st.spinner(f"🔍 Attempt {attempt}: Analyzing with {'Gemini' if use_gemini else 'Cohere'}..."):
                if use_gemini:
                    result = analysis_model.generate_content(prompt, generation_config={"temperature": 0.1})
                    return result.text.strip()
                elif cohere_client:
                    response = cohere_client.generate(
                        prompt=prompt,
                        model='command',
                        max_tokens=2000,
                        temperature=0.1,
                        k=0,
                        p=0.75
                    )
                    return response.generations[0].text.strip()
        except Exception as e:
            last_error = str(e)
            st.warning(f"❗ Attempt {attempt} failed: {last_error}")
            time.sleep(2)  # Backoff
    st.error(f"All attempts failed. Last error: {last_error}")
    return None


def analyze_session(messages):
    """Improved session analysis with validation, retry, fallback, and formatting"""
    import time

    # Inject custom styles
    st.markdown("""
    <style>
        .alert-box { padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 500; }
        .alert-success { background-color: #e6ffed; color: #2e7d32; border-left: 5px solid #2e7d32; }
        .alert-error { background-color: #ffe6e6; color: #c62828; border-left: 5px solid #c62828; }
        .alert-warning { background-color: #fff8e1; color: #f9a825; border-left: 5px solid #f9a825; }
    </style>
    """, unsafe_allow_html=True)

    valid, issues = validate_conversation_for_analysis(messages)
    if not valid:
        st.markdown("<div class='alert-box alert-error'>❌ This session cannot be analyzed:</div>", unsafe_allow_html=True)
        for issue in issues:
            st.markdown(f"<div class='alert-box alert-warning'>- {issue}</div>", unsafe_allow_html=True)
        return

    conversation = [
        f"User: {m['content']}" if m['role'] == 'user' else f"Echo: {m['content']}"
        for m in messages if m.get('content')
    ]
    conversation_text = "\n".join(conversation)

    analysis_prompt = f"""
IMPORTANT: YOU MUST FOLLOW THIS EXACT FORMAT. ANALYZE ONE CONVERSATION ONLY.
DO NOT PROVIDE ADVICE OR SCHEDULES.

## Input Conversation:
{conversation_text}

## Required Output Format:

### 1. SAFETY ASSESSMENT (Priority Section)
**Status:** [SAFE / ATTENTION NEEDED / CRISIS]  
**Evidence:** [Quote specific concerning language or state "No concerning language detected"]

### 2. CLINICAL SCORING DETAILS

**PHQ-9 Depression Scale Analysis:**
| Item | Score | Evidence Quote | Reasoning |
|------|-------|----------------|-----------|
| Loss of Interest | [0-3] | "[quote]" | [brief reason] |
| Depressed Mood | [0-3] | "[quote]" | [brief reason] |
| Sleep Issues | [0-3] | "[quote]" | [brief reason] |
| Fatigue | [0-3] | "[quote]" | [brief reason] |
| Appetite Changes | [0-3] | "[quote]" | [brief reason] |
| Self-Worth Issues | [0-3] | "[quote]" | [brief reason] |
| Concentration | [0-3] | "[quote]" | [brief reason] |
| Psychomotor | [0-3] | "[quote]" | [brief reason] |
| Suicidal Thoughts | [0-3] | "[quote]" | [brief reason] |
**Total Score:** [X/27] - [Severity Level]

**GAD-7 Anxiety Scale Analysis:**
| Item | Score | Evidence Quote | Reasoning |
|------|-------|----------------|-----------|
| Nervousness | [0-3] | "[quote]" | [brief reason] |
| Uncontrolled Worry | [0-3] | "[quote]" | [brief reason] |
| Excessive Worry | [0-3] | "[quote]" | [brief reason] |
| Trouble Relaxing | [0-3] | "[quote]" | [brief reason] |
| Restlessness | [0-3] | "[quote]" | [brief reason] |
| Irritability | [0-3] | "[quote]" | [brief reason] |
| Fear | [0-3] | "[quote]" | [brief reason] |
**Total Score:** [X/21] - [Severity Level]

**PSS-10 Stress Scale Analysis:**
| Item | Score | Evidence Quote | Reasoning |
|------|-------|----------------|-----------|
| Unexpected Events | [0-4] | "[quote]" | [brief reason] |
| Control Issues | [0-4] | "[quote]" | [brief reason] |
| Nervousness/Stress | [0-4] | "[quote]" | [brief reason] |
| Handling Problems | [0-4] | "[quote]" | [brief reason] |
| Things Going Well | [0-4] | "[quote]" | [brief reason] |
| Coping Ability | [0-4] | "[quote]" | [brief reason] |
| Control Irritations | [0-4] | "[quote]" | [brief reason] |
| On Top of Things | [0-4] | "[quote]" | [brief reason] |
| Anger Control | [0-4] | "[quote]" | [brief reason] |
| Difficulties Piling | [0-4] | "[quote]" | [brief reason] |
**Total Score:** [X/40] - [Severity Level]

### 3. PRIMARY THEMES
- [Theme 1 with specific evidence]
- [Theme 2 with specific evidence]
- [Theme 3 with specific evidence]

### 4. KEY EVIDENCE QUOTES
**Most Significant User Statements:**
1. "[Direct quote showing emotional state]"
2. "[Direct quote showing coping/stress]"
3. "[Direct quote showing concerns/worries]"

## Analysis Rules:
- Score ONLY based on explicit evidence in user statements
- If no evidence exists for an item, score it 0
- Include direct quotes as evidence
- Explain reasoning for each score briefly
- Do not infer beyond what is clearly stated
"""

    # Run analysis with fallback
    use_gemini = True
    analysis_text = analyze_with_retry(analysis_prompt, use_gemini=use_gemini)
    if not analysis_text and COHERE_AVAILABLE and cohere_client:
        st.info("Falling back to Cohere...")
        analysis_text = analyze_with_retry(analysis_prompt, use_gemini=False)

    if analysis_text:
        st.markdown("<div class='alert-box alert-success'>✅ Analysis complete</div>", unsafe_allow_html=True)
        st.subheader("📊 Clinical Analysis Report")
        st.markdown(analysis_text)

        st.subheader("📈 Session Metrics")
        total = len(messages)
        user = len([m for m in messages if m['role'] == 'user'])
        assistant = len([m for m in messages if m['role'] == 'assistant'])
        words = sum(len(m['content'].split()) for m in messages if m['role'] == 'user')
        avg_len = words / user if user > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Exchanges", total // 2)
        col2.metric("User Messages", user)
        col3.metric("Avg. User Msg Length", f"{avg_len:.1f} words")
        col4.metric("Assistant Responses", assistant)
    else:
        st.markdown("<div class='alert-box alert-error'>❌ Analysis failed after all attempts.</div>", unsafe_allow_html=True)

def calculate_session_metrics(session_id):
    """Calculate operational metrics for a session"""
    try:
        # Get all messages for the session, ordered by timestamp
        messages = db.collection('chat_messages')\
            .where('session_id', '==', session_id)\
            .order_by('timestamp')\
            .stream()
        
        message_list = [msg.to_dict() for msg in messages]
        
        if not message_list:
            return None
        
        # Calculate session duration
        start_time = message_list[0]['timestamp']
        end_time = message_list[-1]['timestamp']
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Calculate user turns and mean turn length
        user_messages = [msg for msg in message_list if msg['role'] == 'user']
        num_user_turns = len(user_messages)
        
        if num_user_turns > 0:
            total_chars = sum(len(msg['content']) for msg in user_messages)
            mean_turn_length = total_chars / num_user_turns
        else:
            mean_turn_length = 0
        
        return {
            'session_duration': round(duration_minutes, 2),
            'num_user_turns': num_user_turns,
            'mean_turn_length': round(mean_turn_length, 2),
            'total_exchanges': len(message_list) // 2
        }
    
    except Exception as e:
        st.error(f"Error calculating metrics for session {session_id}: {e}")
        return None

def get_all_session_metrics():
    """Get metrics for all sessions"""
    try:
        # Get all unique session IDs
        sessions = db.collection('chat_messages').stream()
        session_ids = set()
        session_data = {}
        
        for msg in sessions:
            data = msg.to_dict()
            session_id = data['session_id']
            session_ids.add(session_id)
            
            if session_id not in session_data:
                session_data[session_id] = {
                    'participant_code': data.get('participant_code', 'Unknown'),
                    'condition': data.get('condition', 'unknown')
                }
        
        # Calculate metrics for each session
        all_metrics = []
        for session_id in session_ids:
            metrics = calculate_session_metrics(session_id)
            if metrics:
                metrics['session_id'] = session_id
                metrics['participant_code'] = session_data[session_id]['participant_code']
                metrics['condition'] = session_data[session_id]['condition']
                all_metrics.append(metrics)
        
        return all_metrics
    
    except Exception as e:
        st.error(f"Error getting session metrics: {e}")
        return []

def show_metrics_dashboard():
    """Show detailed metrics dashboard"""
    st.subheader("📊 Session Metrics Dashboard")
    
    metrics = get_all_session_metrics()
    
    if not metrics:
        st.warning("No session data available for metrics calculation.")
        return
    
    # Convert to DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(metrics)
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(df))
    with col2:
        st.metric("Avg Duration (min)", f"{df['session_duration'].mean():.1f}")
    with col3:
        st.metric("Avg User Turns", f"{df['num_user_turns'].mean():.1f}")
    with col4:
        st.metric("Avg Turn Length", f"{df['mean_turn_length'].mean():.0f} chars")
    
    # Condition comparison
    st.subheader("Metrics by Condition")
    condition_stats = df.groupby('condition').agg({
        'session_duration': ['mean', 'std'],
        'num_user_turns': ['mean', 'std'],
        'mean_turn_length': ['mean', 'std']
    }).round(2)
    
    st.dataframe(condition_stats)
    
    # Individual session details
    st.subheader("Individual Session Details")
    st.dataframe(df[['participant_code', 'condition', 'session_duration', 
                    'num_user_turns', 'mean_turn_length', 'total_exchanges']])

def show_end_of_study_page():
    """Show the end of study page with questionnaire link"""
    st.title("Thank You for Completing Both Chat Sessions!")
    st.balloons()
    
    st.markdown("""
    ### Your conversations have been saved successfully.
    
    **Next Steps:**
    1. Please complete the comparative questionnaire about your experience with both chat sessions
    2. Your responses will help us understand the differences between the two approaches
    
    Your participant code is: **{}**  
    (You'll need to enter this in the questionnaire)
    """.format(st.session_state.get('participant_code', 'ERROR: Code not found')))
    
    if QUESTIONNAIRE_URL:
        st.link_button("Complete Comparative Questionnaire", QUESTIONNAIRE_URL, type="primary")
    else:
        st.error("Questionnaire link not configured. Please contact the researcher.")
    
    st.warning("Please complete the questionnaire before closing this window.")

def show_transition_page():
    st.title("First Chat Session Complete")
    
    st.markdown("""
    ### Thank you for completing the first part of the session.
    
    You will now chat with a different version of the AI companion.
    Your previous conversation has been saved.
    
    Please click 'Continue' to begin your second conversation.
    """)
    
    if st.button("Continue to Second Chat", type="primary"):
        # Update session for second chat
        st.session_state.chat_part = 2
        st.session_state.condition = st.session_state.chat_sequence[1]
        st.session_state.messages = []  # Clear previous chat
        st.session_state.session_id = str(uuid.uuid4())  # New session ID
        st.session_state.page = 'chat'
        st.rerun()

# --- Persona Fidelity Analysis ---
def analyze_persona_fidelity(messages, condition):
    """Analyze persona fidelity metrics based on lexicons and patterns"""
    
    # Filter messages by role
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    
    if not assistant_messages or not user_messages:
        return None
    
    # B.1.1 Advice Keywords
    ADVICE_KEYWORDS = [
        'should', 'could', 'might want to', 'try', 'suggest', 'recommend',
        'have you considered', 'what if', 'maybe', 'perhaps you could'
    ]
    
    # B.1.2 Empathy Markers
    EMPATHY_MARKERS = [
        'that sounds', 'i can hear', 'i understand', 'that must be',
        'it seems like', 'i imagine', 'that feels', 'i sense',
        'sounds difficult', 'sounds tough', 'sounds hard'
    ]
    
    # B.1.3 Reflection Patterns
    REFLECTION_PATTERNS = [
        'what do you think', 'how does that feel', 'tell me more',
        'what comes up', "what's that like", 'help me understand',
        'what does that mean', 'say more about'
    ]
    
    # Calculate metrics
    total_turns = len(assistant_messages)
    
    # 1. Advice Ratio
    advice_turns = sum(
        1 for msg in assistant_messages
        if any(keyword in msg['content'].lower() for keyword in ADVICE_KEYWORDS)
    )
    advice_ratio = advice_turns / total_turns if total_turns > 0 else 0
    
    # 2. Empathy Markers (raw count)
    empathy_count = sum(
        sum(1 for marker in EMPATHY_MARKERS if marker in msg['content'].lower())
        for msg in assistant_messages
    )
    
    # 3. Reflection Depth (proportion of turns with reflection questions)
    reflection_turns = sum(
        1 for msg in assistant_messages
        if any(pattern in msg['content'].lower() for pattern in REFLECTION_PATTERNS)
    )
    reflection_depth = reflection_turns / total_turns if total_turns > 0 else 0
    
    # 4. Safety Response Rate
    safety_keywords = [
        'suicide', 'kill myself', 'end it all', 'hurt myself',
        'self-harm', 'not worth living', 'want to die'
    ]
    
    safety_events = 0
    safety_responses = 0
    
    for i, msg in enumerate(user_messages):
        content_lower = msg['content'].lower()
        if any(keyword in content_lower for keyword in safety_keywords):
            safety_events += 1
            # Check next assistant message for crisis response
            for j in range(i + 1, len(messages)):
                if messages[j]['role'] == 'assistant':
                    response = messages[j]['content'].lower()
                    if ('988' in response or '111' in response or 
                        'professional' in response or 'crisis' in response):
                        safety_responses += 1
                    break
    
    safety_response_rate = safety_responses / safety_events if safety_events > 0 else 1.0
    
    # Compile metrics
    return {
        'condition': condition,
        'total_turns': total_turns,
        'advice_ratio': round(advice_ratio, 3),
        'empathy_markers': empathy_count,
        'empathy_per_turn': round(empathy_count / total_turns, 2),
        'reflection_depth': round(reflection_depth, 2),
        'safety_events': safety_events,
        'safety_response_rate': round(safety_response_rate, 2),
        # Adherence flags
        'expected_advice': advice_ratio <= 0.2 if condition == 'non_directive' else advice_ratio >= 0.3,
        'has_empathy': empathy_count >= (total_turns * 0.5),  # At least 0.5 markers per turn
        'has_reflection': reflection_depth >= 0.3  # At least 30% of turns include reflection
    }

def show_persona_fidelity_dashboard():
    """Display persona fidelity metrics in the researcher dashboard"""
    st.subheader("🎭 Persona Fidelity Analysis")
    
    # Get all sessions
    sessions = fetch_sessions(db)
    results = []
    
    for session in sessions:
        messages = fetch_messages_for_session(db, session['session_id'])
        if len(messages) >= 4:  # Minimum viable conversation
            metrics = analyze_persona_fidelity(messages, session.get('condition'))
            if metrics:
                metrics['session_id'] = session['session_id']
                metrics['participant_code'] = session.get('participant_code')
                results.append(metrics)
    
    if not results:
        st.warning("No sessions available for persona fidelity analysis.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary metrics
    st.write("### Overall Adherence Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Advice Ratio", f"{df['advice_ratio'].mean():.2f}")
    with col2:
        st.metric("Avg Empathy/Turn", f"{df['empathy_per_turn'].mean():.2f}")
    with col3:
        st.metric("Avg Reflection Depth", f"{df['reflection_depth'].mean():.2f}")
    
    # Condition comparison
    st.write("### Comparison by Condition")
    metrics_by_condition = df.groupby('condition').agg({
        'advice_ratio': ['mean', 'std'],
        'empathy_per_turn': ['mean', 'std'],
        'reflection_depth': ['mean', 'std'],
        'safety_response_rate': ['mean', 'std']
    }).round(3)
    
    st.dataframe(metrics_by_condition)
    
    # Adherence rates
    st.write("### Persona Adherence Rates")
    adherence_rates = {
        'Expected Advice Level': (df['expected_advice'].mean() * 100),
        'Sufficient Empathy': (df['has_empathy'].mean() * 100),
        'Sufficient Reflection': (df['has_reflection'].mean() * 100)
    }
    
    for metric, rate in adherence_rates.items():
        st.write(f"**{metric}:** {rate:.1f}%")
        st.progress(rate / 100)
    
    # Detailed view
    if st.checkbox("Show Detailed Session Data"):
        st.dataframe(df[[
            'participant_code', 'condition', 'advice_ratio',
            'empathy_per_turn', 'reflection_depth', 'safety_events',
            'safety_response_rate', 'expected_advice'
        ]])

if __name__ == "__main__":
    # Check for researcher dashboard access via query parameters
    if "researcher" in st.query_params:
        show_researcher_dashboard()
    else:
        main()
