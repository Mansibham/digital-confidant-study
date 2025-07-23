import streamlit as st
import google.generativeai as genai
import google.cloud.firestore as firestore
import json
from datetime import datetime
import uuid
import hashlib
import requests

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
        "system_prompt": """You are Echo, a gentle and empathetic AI companion practicing person-centered therapy principles:
        Your core identity is that of a passive listener. You MUST follow these rules without exception:

        1.  **VARY YOUR RESPONSES:** This is a critical rule. Do not repeat the same reflective question or validation phrase multiple times. Use a wide variety of phrasings to keep the conversation feeling natural and not robotic.
        2.  **ABSOLUTELY NO ADVICE:** You are strictly prohibited from giving any advice, suggestions, solutions, or coping strategies, no matter how simple.
        3.  **REFLECT AND VALIDATE:** Your primary function is to reflect the user's feelings and validate their experience. Use phrases like "That sounds incredibly difficult," "I hear how much that's weighing on you," or "It makes sense that you would feel that way."
        4.  **ASK OPEN-ENDED QUESTIONS:** Encourage the user to explore their own feelings with questions like "How does that feel for you?" or "What's on your mind as you think about that?"
        5.  **HANDLE ADVICE-SEEKING:** If the user directly asks "What should I do?", you MUST deflect. Your response should be a variation of: "That's a really important question, and it shows you're thinking deeply about this. I can't tell you what to do, but I am here to listen as you work through your own thoughts and find the path that feels right for you."
        6.  **CRITICAL SAFETY RULE:** If a user expresses any intent of self-harm or being in immediate danger, your ONLY response must be: "It sounds like you are in a great deal of pain, and it's incredibly brave of you to share that. For immediate support, it's very important to talk to a trained professional. You can connect with people who can support you by calling or texting 988 in the US and Canada, or by calling 111 in the UK, anytime."
        """,
        "description": "Non-directive: Empathetic listening and reflection"
    },
    "directive": {
        "name": "Echo (Guide)",
        "system_prompt": """You are Echo, a supportive AI companion that provides practical guidance and coping strategies:
        - Listen empathetically and acknowledge feelings.
        - Offer one single, gentle, evidence-based coping suggestion when appropriate.
        - Frame suggestions as options, not commands (e.g., "Some people find...").
        - Keep suggestions general and safe (e.g., walking, journaling, deep breathing).
        - After offering a suggestion, immediately return to a listening role.
        - When offering a suggestion, briefly mention its purpose. For example, instead of just "try journaling," say "try journaling, as it can be a great way to organize and understand your thoughts.
        - **CRITICAL SAFETY RULE:** You are a supportive companion, not a crisis service. If a user expresses any intent of self-harm or being in immediate danger, your ONLY response must be: "It sounds like you are in a great deal of pain, and it's incredibly brave of you to share that. For immediate support, it's very important to talk to a trained professional. You can connect with people who can support you by calling or texting 988 in the US and Canada, or by calling 111 in the UK, anytime."
        """,
        "description": "Directive: Practical coping suggestions and guidance"
    }
}

# --- Styling ---
def inject_css():
    st.markdown("""
    <style>
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

# --- Main App Logic ---
QUESTIONNAIRE_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdvBnxXnCvLcNLKYEBFk-c1Bi_cAGjv3GpxjV_sxCrUpKbUng/viewform?usp=header"

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
    elif st.session_state.page == 'participant_code':  # Add this page
        show_participant_code_page()
    elif st.session_state.page == 'chat':
        show_chat_interface()
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
    
    code = st.text_input("Participant Code", placeholder="e.g., MH-001-A")
    
    if st.button("Start Chat", type="primary") and code:
        # Initialize session with participant code
        st.session_state.participant_code = code
        st.session_state.condition = get_condition_from_code(code)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        
        # Save initial session data
        try:
            db.collection('chat_sessions').document(st.session_state.session_id).set({
                'participant_code': code,
                'condition': st.session_state.condition,
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
                    'ai_provider': data.get('ai_provider', 'unknown'),  # Set default value
                    'condition': data.get('condition', 'unknown')  # Set default value
                }
            
            session_data[session_id]['messages'].append(data)
            if data.get('timestamp'):
                session_data[session_id]['last_time'] = data.get('timestamp')
        
        st.write(f"**Total Sessions:** {len(session_data)}")
        
        # Provider stats with null checking
        provider_stats = {}
        condition_stats = {}
        for session_id, data in session_data.items():
            provider = data.get('ai_provider', 'unknown')  # Get with default value
            condition = data.get('condition', 'unknown')   # Get with default value
            
            # Handle None values
            provider = provider if provider else 'unknown'
            condition = condition if condition else 'unknown'
            
            provider_stats[provider] = provider_stats.get(provider, 0) + 1
            condition_stats[condition] = condition_stats.get(condition, 0) + 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**AI Provider Distribution:**")
            for provider, count in provider_stats.items():
                st.write(f"- {provider if provider else 'unknown'}: {count} sessions")
        
        with col2:
            st.write("**Condition Distribution:**")
            for condition, count in condition_stats.items():
                st.write(f"- {condition if condition else 'unknown'}: {count} sessions")
        
        # Session details
        for session_id, data in session_data.items():
            with st.expander(f"Session: {data['participant_code']} ({len(data['messages'])} messages) - {data['ai_provider'].title()} - {data['condition'].title()}"):
                
                # Show conversation
                st.write("**Conversation:**")
                for msg in data['messages']:
                    role = "🧑 User" if msg['role'] == 'user' else "🤖 Echo"
                    st.write(f"{role}: {msg['content']}")
                
                # Analysis button
                if st.button(f"Analyze Session {session_id[:8]}...", key=f"analyze_{session_id}"):
                    analyze_session(data['messages'])
    
    except Exception as e:
        st.error(f"Failed to load session data: {e}")

def analyze_session(messages):
    """Session analysis using Gemini with Cohere fallback"""
    
    # Build conversation text
    conversation = []
    for msg in messages:
        if msg['role'] == 'user':
            conversation.append(f"User: {msg['content']}")
        else:
            conversation.append(f"Echo: {msg['content']}")
    
    conversation_text = "\n".join(conversation)
    
    analysis_prompt = f"""
    Role
    You are an expert psychological analyst AI specialized in clinical assessment of therapeutic conversations. You provide structured, evidence-based analysis using standardized clinical scales to support mental health research.
    Task
    Your task is to review conversation transcripts between users and the Digital Confidant chatbot and generate a highly structured, clean, and readable clinical summary based ONLY on the user's statements.
    Input
    You will receive complete chat session transcripts containing conversations between users and the therapeutic chatbot, including timestamps and participant information.
    Output
    Your entire output MUST be formatted in clean Markdown. You will prioritize information by putting the most critical alerts first. All detailed scale breakdowns MUST be presented in tables.
    Analyze the provided transcript and generate a report with the following sections in this exact order:
    1. Red Flag Identification
    Scan the text for any language that suggests immediate risk, such as self-harm, suicidal ideation, or severe distress. This section must be your top priority.
    State either "No red flags identified." or "IMMEDIATE RED FLAG DETECTED:" followed by the concerning quote.
    2. Clinical Summary at a Glance
    Provide the total scores and clinical interpretation for each assessment. This gives an immediate overview of the user's state.
    PHQ-9 (Depression): [Total Score] / 27 (Interpretation: e.g., Mild, Moderately Severe)
    GAD-7 (Anxiety): [Total Score] / 21 (Interpretation: e.g., Mild, Severe)
    PSS-10 (Perceived Stress): [Total Score] / 40 (Interpretation: e.g., Low, High)
    3. Key Qualitative Themes
    In 2-4 bullet points, identify the primary narrative themes or stressors mentioned by the user. This provides the context behind the scores.

    [Theme 1, e.g., Significant job-related stress from management]
    [Theme 2, e.g., Expressed financial insecurity and anxiety about the future]
    [Theme 3, e.g., Feelings of being "stuck" or lacking control]

    4. Detailed Assessment Breakdown
    Present the item-by-item breakdown for each scale in a clean table format for easy review.
    PHQ-9 Assessment (Depression)
    ItemScore (0-3)Justification (Evidence from Transcript)Interest/Pleasure[Score][Quote or summary of evidence]Feeling Down/Hopeless[Score][Quote or summary of evidence]Sleep Issues[Score][Quote or summary of evidence]Tired/Little Energy[Score][Quote or summary of evidence]Appetite Issues[Score][Quote or summary of evidence]Feeling Bad About Self[Score][Quote or summary of evidence]Trouble Concentrating[Score][Quote or summary of evidence]Slow/Restless[Score][Quote or summary of evidence]Self-Harm Thoughts[Score][Quote or summary of evidence]
    GAD-7 Assessment (Anxiety)
    ItemScore (0-3)Justification (Evidence from Transcript)Nervous/Anxious[Score][Quote or summary of evidence]Uncontrollable Worry[Score][Quote or summary of evidence]Worrying Too Much[Score][Quote or summary of evidence]Trouble Relaxing[Score][Quote or summary of evidence]Restlessness[Score][Quote or summary of evidence]Easily Annoyed[Score][Quote or summary of evidence]Feeling Afraid[Score][Quote or summary of evidence]
    PSS-10 Assessment (Perceived Stress)
    ItemScore (0-4)Justification (Evidence from Transcript)Upset by Unexpected[Score][Quote or summary of evidence]Unable to Control[Score][Quote or summary of evidence]Felt Nervous/Stressed[Score][Quote or summary of evidence]Lacked Confidence[Score][Quote or summary of evidence]Things Not Going Way[Score][Quote or summary of evidence]Could Not Cope[Score][Quote or summary of evidence]Unable to Control Irritations[Score][Quote or summary of evidence]Not on Top of Things[Score][Quote or summary of evidence]Angered by Uncontrollables[Score][Quote or summary of evidence]Difficulties Piling Up[Score][Quote or summary of evidence]
    Constraints
    DO NOT:

    Invent any information not present in the transcript
    Provide clinical diagnoses or medical advice
    Include personal identifying information
    Make assumptions beyond what is directly stated or strongly implied

    ALWAYS:

    Base all analysis on direct or strong indirect evidence from the text
    If no evidence is present for an item, score it 0 and state "No evidence found"
    Use actual quotes from the transcript whenever possible
    Prioritize safety by identifying red flags first

    Capabilities
    You can:

    Clinical Scale Assessment: Systematically evaluate PHQ-9, GAD-7, and PSS-10 indicators
    Evidence-Based Scoring: Assign scores based on specific conversation content
    Risk Detection: Identify concerning language requiring immediate attention
    Thematic Analysis: Extract key narrative themes from user statements
    Structured Reporting: Generate clean, readable clinical summaries

    Reminders

    Evidence-Based Only: Every score must be justified with specific evidence from the transcript
    Safety First: Always check for and prioritize red flag identification
    Use Clinical Criteria: Apply standard scoring guidelines for each assessment scale
    Stay Objective: Report only what is observable in the conversation
    Maintain Structure: Follow the exact format and section order specified
    Quote When Possible: Use direct quotes to support your assessments
    Score Conservatively: When in doubt, use lower scores rather than inflating them
    Your response MUST be a clean, structured Markdown report with the sections and tables as specified.
    
    **Conversation:**
    {conversation_text}
    """
    
    try:
        # Let researcher choose the primary provider
        analysis_provider = st.radio(
            "Select Analysis Provider",
            ["Gemini", "Cohere"] if COHERE_AVAILABLE and cohere_client else ["Gemini"],
            help="Choose which AI model to use for analysis"
        )
        
        with st.spinner(f"🔍 Analyzing session with {analysis_provider}..."):
            try:
                if analysis_provider == "Gemini":
                    analysis = analysis_model.generate_content(analysis_prompt)
                    analysis_text = analysis.text
            except Exception as e:
                st.error(f"Gemini Analysis Error: {str(e)}")
                if COHERE_AVAILABLE and cohere_client:
                    st.info("Falling back to Cohere...")
                    try:
                        response = cohere_client.generate(
                            prompt=analysis_prompt,
                            model='command',
                            max_tokens=2000,
                            temperature=0.7,
                            k=0,
                            p=0.75
                        )
                        analysis_text = response.generations[0].text
                    except Exception as cohere_error:
                        st.error(f"Cohere fallback failed: {str(cohere_error)}")
                        return
                else:
                    st.error("No fallback option available")
                    return
            
            # Display results
            st.subheader("📊 Detailed Analysis Results")
            st.markdown(analysis_text)
            
            # Quick metrics
            st.subheader("📈 Session Metrics")
            total_messages = len(messages)
            user_messages = len([m for m in messages if m['role'] == 'user'])
            assistant_messages = len([m for m in messages if m['role'] == 'assistant'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Exchanges", total_messages // 2)
            with col2:
                st.metric("User Messages", user_messages)
            with col3:
                st.metric("Assistant Responses", assistant_messages)
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.info("Please try again or contact support if the problem persists.")

def show_end_of_study_page():
    """Show the end of study page with questionnaire link"""
    st.title("Thank You for Participating!")
    st.balloons()
    
    st.markdown("""
    ### Your conversation has been saved successfully.
    
    **Next Steps:**
    1. Please complete a brief questionnaire about your experience
    2. Your responses will help us improve the MindReflect Chatbot
    
    Your participant code is: **{}**  
    (You'll need to enter this in the questionnaire)
    """.format(st.session_state.get('participant_code', 'ERROR: Code not found')))
    
    if QUESTIONNAIRE_URL:
        st.link_button("Complete Final Questionnaire", QUESTIONNAIRE_URL, type="primary")
    else:
        st.error("Questionnaire link not configured. Please contact the researcher.")
    
    st.warning("Please complete the questionnaire before closing this window.")

# --- App Entry Point ---
if __name__ == "__main__":
    # Check for researcher mode
    if st.query_params.get("researcher") == "true":
        show_researcher_dashboard()
    else:
        main()
