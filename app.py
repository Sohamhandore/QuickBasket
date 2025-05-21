import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import re
import os

# Set page config
st.set_page_config(
    page_title="Quick Basket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap');

    /* Root Layout */
    .main {
        padding: 2rem;
        background-color: #FFF8E7;
        font-family: 'Poppins', sans-serif;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #4A4A4A;
    }

    .title-text {
        color: #D4A017;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(212, 160, 23, 0.3);
        animation: floatText 5s ease-in-out infinite;
        font-family: 'Montserrat', sans-serif;
    }

    .subtitle-text {
        color: #8B7355;
        font-size: 1.4rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
        animation: typing 4s steps(40, end), blink-caret 0.75s step-end infinite;
        white-space: nowrap;
        overflow: hidden;
        border-right: 3px solid #D4A017;
        font-family: 'Poppins', sans-serif;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 15px 25px;
        color: #4A4A4A;
        background-color: #FFF8E7;
        border: 2px solid #D4A017;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease-in-out;
    }

    .stTextInput > div > div > input:hover {
        border-color: #8B7355;
        box-shadow: 0 0 15px rgba(139, 115, 85, 0.3);
    }

    .stTextInput > div > div > input:focus {
        border-color: #8B7355;
        box-shadow: 0 0 20px rgba(139, 115, 85, 0.4);
    }

    /* Chat Messages */
    .chat-message {
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        color: #4A4A4A;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .chat-message:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }

    .chat-message.user {
        background: linear-gradient(135deg, #FFF8E7 0%, #F5E6D3 100%);
        border-left: 5px solid #D4A017;
    }

    .chat-message.assistant {
        background: linear-gradient(135deg, #F5E6D3 0%, #E6D5C3 100%);
        border-left: 5px solid #8B7355;
    }

    .chat-message .content {
        margin-top: 0.8rem;
        color: #4A4A4A;
        animation: fadeInUp 0.5s ease-in-out;
    }

    .chat-message strong {
        color: #D4A017;
        font-weight: 600;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #FFF8E7 !important;
        color: #4A4A4A !important;
        font-family: 'Poppins', sans-serif;
    }

    .sidebar h2 {
        color: #D4A017;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .sidebar p {
        color: #8B7355;
        font-size: 1.1rem;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 25px;
        padding: 0.8rem 1.5rem;
        background: linear-gradient(135deg, #D4A017 0%, #B38B0B 100%);
        color: #FFF8E7;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #8B7355 0%, #6B5B4B 100%);
        color: #FFF8E7;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139, 115, 85, 0.3);
    }

    /* Markdown Styling */
    .stMarkdown {
        color: #4A4A4A;
        font-family: 'Poppins', sans-serif;
        line-height: 1.6;
    }

    .stMarkdown p {
        color: #4A4A4A;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #D4A017;
        font-family: 'Montserrat', sans-serif;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .stMarkdown strong {
        color: #8B7355;
        font-weight: 600;
    }

    .stMarkdown code {
        background-color: #F5E6D3;
        color: #D4A017;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Consolas', monospace;
    }

    /* List Styling */
    .stMarkdown ul li {
        color: #8B7355;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        list-style-type: none;
        padding-left: 1.5rem;
        position: relative;
    }

    .stMarkdown ul li:before {
        content: "•";
        color: #D4A017;
        position: absolute;
        left: 0;
        font-size: 1.2rem;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 20px, 0);
        }
        to {
            opacity: 1;
            transform: none;
        }
    }

    @keyframes floatText {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #D4A017; }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .title-text {
            font-size: 2rem;
        }
        .subtitle-text {
            font-size: 1.1rem;
        }
        .chat-message {
            padding: 1.2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    if 'context' not in st.session_state:
        st.session_state.context = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""
    if 'last_intent' not in st.session_state:
        st.session_state.last_intent = None
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {}
    if 'response_count' not in st.session_state:
        st.session_state.response_count = {}
    if 'product_context' not in st.session_state:
        st.session_state.product_context = None
    if 'refund_context' not in st.session_state:
        st.session_state.refund_context = None

# Load training data
def load_data():
    try:
        df = pd.read_csv('mock_intents.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Initialize model components
def init_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        return tokenizer, model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None, None

# Train classifier
def train_model(df, tokenizer, model):
    try:
        texts = df['User_Query'].values
        labels = df['Intent'].values
        
        # Using TF-IDF with bigrams for better feature extraction
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        classifier = MultinomialNB(alpha=0.1)
        
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier)
        ])
        
        pipeline.fit(texts, labels)
        return pipeline
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Predict intent from user input
def get_intent(text, classifier):
    try:
        # Check for policy-related keywords
        policy_keywords = ['policy', 'rules', 'terms', 'conditions']
        if any(keyword in text.lower() for keyword in policy_keywords):
            if 'refund' in text.lower() or 'return' in text.lower():
                return "Return Policy", 0.9
            return "Unknown/Other", 0.8
            
        # Check for refund-related keywords
        refund_keywords = ['refund', 'return', 'money back', 'cancel']
        if any(keyword in text.lower() for keyword in refund_keywords):
            if 'policy' in text.lower():
                return "Return Policy", 0.9
            return "Return & Refund Policy", 0.9
            
        # Check for product availability keywords
        availability_keywords = ['in stock', 'available', 'have', 'sell', 'carry', 'want']
        product_keywords = ['nike', 'adidas', 'puma', 'shoes', 'clothes', 'apparel', 'sneakers', 'sportswear']
        
        if any(keyword in text.lower() for keyword in availability_keywords):
            if any(product in text.lower() for product in product_keywords):
                return "Product Availability", 0.9
            return "Unknown/Other", 0.8
            
        # Check for order-related keywords
        if 'order' in text.lower() or 'ordered' in text.lower():
            return "Order Tracking", 0.8
            
        # Check for question words that might indicate unknown queries
        question_words = ['who', 'what', 'when', 'where', 'why', 'how']
        if any(word in text.lower().split() for word in question_words):
            # If it's a question about something not in our domain
            if not any(keyword in text.lower() for keyword in ['order', 'return', 'product', 'store', 'delivery', 'refund', 'shipping']):
                return "Unknown/Other", 1.0
        
        intent = classifier.predict([text])[0]
        probs = classifier.predict_proba([text])[0]
        confidence = probs[classifier.classes_ == intent][0]
        
        # If confidence is too low, treat as unknown
        if confidence < 0.3:
            return "Unknown/Other", confidence
            
        return intent, confidence
    except Exception as e:
        st.error(f"Error predicting intent: {str(e)}")
        return "Unknown/Other", 0.0

# Response templates
RESPONSE_TEMPLATES = {
    'Order Tracking': [
        "Please share your order ID to help track your package.",
        "I can help you track your order. Could you provide your order ID?",
        "To track your order, I'll need your order ID. Could you share that?"
    ],
    'Return & Refund Policy': [
        "I can help you with the refund process. Could you please provide your order ID?",
        "I'll help you process your refund. First, I'll need your order ID.",
        "To process your refund, I'll need your order ID. Could you share that?"
    ],
    'Return Policy': [
        "Our return policy allows returns within 30 days of delivery. Items must be unused and in original packaging. Would you like to know more about the return process?",
        "You can return items within 30 days of delivery for a full refund. The item must be in its original condition. Would you like to start the return process?",
        "We offer a 30-day return window with full refunds. The product must be unused and in its original packaging. Would you like to know more about how to return an item?"
    ],
    'Product Availability': [
        "Let me check our inventory for {product} products. Would you like to know about specific models or sizes?",
        "We carry {product} products in our stores. What specific style or size are you looking for?",
        "Yes, we have {product} items available. Would you like to know about our current collection or check a specific store's inventory?"
    ],
    'Store Location/Hours': [
        "Let me help you find the nearest store. Could you please share your location? Our stores are open Monday-Saturday, 9 AM to 9 PM.",
        "I can help you locate our nearest store. Which area are you in? All our stores are open Monday-Saturday, 9 AM to 9 PM.",
        "To find the closest store to you, please let me know your location. Our stores operate Monday-Saturday, 9 AM to 9 PM."
    ],
    'General Greetings': [
        "Hello! How can I assist you today?",
        "Hi there! How may I help you?",
        "Welcome! What can I do for you today?"
    ],
    'Unknown/Other': [
        "I can help you with information about our sports apparel and footwear collection, including Nike, Adidas, Puma, and other brands. What would you like to know?",
        "I can assist you with product availability, orders, returns, and store information. What specific information are you looking for?",
        "I can help you with questions about our products, orders, returns, and store locations. How may I assist you today?"
    ]
}

# Store location responses
STORE_LOCATION_RESPONSES = [
    "Here are our store locations:\n1. City Center: 123 Main Street (Search 'ShopFast City Center' on Google Maps)\n2. Downtown: 456 Market Square (Accessible by public transport)\n3. Central Station: 789 Shopping Plaza (Ample parking available)\nCould you share your location so I can help you find the nearest one?",
    "We have three convenient locations:\n1. City Center (123 Main Street)\n2. Downtown (456 Market Square)\n3. Near Central Station (789 Shopping Plaza)\nWhich area are you in? I can help you find the closest store.",
    "Our stores are located at:\n1. 123 Main Street, City Center (Easy to find on Google Maps)\n2. 456 Market Square, Downtown (Great public transport access)\n3. 789 Shopping Plaza, near Central Station (Free parking available)\nLet me know your location to find the nearest store."
]

# Generate appropriate response
def get_response(intent, confidence, context=None, last_intent=None, user_input=None):
    try:
        # Initialize response count for this intent if not exists
        if intent not in st.session_state.response_count:
            st.session_state.response_count[intent] = 0
        
        # Reset refund context if asking about policy
        if 'policy' in user_input.lower():
            st.session_state.refund_context = None
        
        # Handle store location queries
        if any(keyword in user_input.lower() for keyword in ['store', 'location', 'nearest', 'where']):
            return np.random.choice(STORE_LOCATION_RESPONSES)
        
        # Get base response
        responses = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES['Unknown/Other'])
        
        # If we've used this intent too many times in a row, force Unknown/Other
        if st.session_state.response_count[intent] >= 2:
            intent = "Unknown/Other"
            responses = RESPONSE_TEMPLATES['Unknown/Other']
            st.session_state.response_count = {}  # Reset counts
        
        response = np.random.choice(responses)
        st.session_state.response_count[intent] += 1
        
        # Handle product availability responses
        if intent == "Product Availability":
            product = get_context(user_input)
            if product:
                response = response.format(product=product.capitalize())
            else:
                response = "I can help you check product availability. Which brand or item are you looking for? We carry Nike, Adidas, Puma, and other sports brands."
        
        # Add context if available and not already handled
        elif context and intent in ['Order Tracking', 'Return & Refund Policy']:
            response = f"Regarding your {context} purchase, {response.lower()}"
        
        # Handle refund follow-ups only if not asking about policy
        if ('refund' in user_input.lower() or (last_intent == 'Return & Refund Policy' and 'refund' in user_input.lower())) and 'policy' not in user_input.lower():
            response = "I'll help you process your refund. Could you please provide your order ID? Once you share that, I can guide you through the refund process."
            st.session_state.refund_context = True
        
        # Add confidence score for low confidence predictions
        if confidence < 0.5:
            response += f"\n\n(Confidence: {confidence:.2%})"
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Could you please try rephrasing your question?"

# Extract product context from message
def get_context(message):
    keywords = ['nike', 'adidas', 'puma', 'shoes', 'sneakers', 'sportswear', 'clothes', 'apparel', 'watch', 'accessories']
    for keyword in keywords:
        if keyword.lower() in message.lower():
            st.session_state.product_context = keyword
            return keyword
    return None

def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #2196F3;'>Quick Basket  Assistant</h2>
                <p style='color: #666;'>AI That Shops with You</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🤖 Capabilities")
        st.markdown("""
            - 📦 Track my Order
            - 🔄 Return or get a Refund
            - 🏪 About store information
            - 🛍️ Check product stock
            - 👋 Help and Assistance
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.context = None
            st.session_state.last_input = ""
            st.session_state.last_intent = None
            st.session_state.conversation_context = {}
            st.session_state.response_count = {}
            st.session_state.product_context = None
            st.session_state.refund_context = None
            st.rerun()
    
    # Main content
    st.markdown("<h1 class='title-text'>Quick Basket  Customer Support</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Your Fast, Friendly, and Frugal shopping assistant is here to help!</p>", unsafe_allow_html=True)
    
    # Model initialization
    if st.session_state.model is None:
        with st.spinner("🔄 Loading and training the model... plz wait"):
            df = load_data()
            if df is not None:
                tokenizer, model = init_model()
                if tokenizer is not None and model is not None:
                    classifier = train_model(df, tokenizer, model)
                    if classifier is not None:
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.classifier = classifier
                        st.success("✅ Training of the model completed without issues!")
                    else:
                        st.error("❌ Failed to train the classifier.")
                else:
                    st.error("❌ Failed to initialize the model.")
            else:
                st.error("❌ Failed to load the training data.")
    
    # Chat interface
    st.markdown("### 💬 Chat with our Quick AI assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class='chat-message user'>
                    <div class='content'>
                        <strong>You:</strong> {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message assistant'>
                    <div class='content'>
                        <strong>Assistant:</strong> {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Handle user input
    user_input = st.text_input("Type your message:", key="user_input", placeholder="Ask me anything about our products, orders, or stores...")
    
    if user_input and user_input != st.session_state.last_input and st.session_state.classifier is not None:
        # Update session state
        st.session_state.last_input = user_input
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Process message
        context = get_context(user_input)
        if context:
            st.session_state.context = context
        
        intent, confidence = get_intent(user_input, st.session_state.classifier)
        response = get_response(intent, confidence, st.session_state.context, st.session_state.last_intent, user_input)
        
        # Update conversation
        st.session_state.last_intent = intent
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        
        # Refresh UI
        st.rerun()

if __name__ == "__main__":
    main() 
