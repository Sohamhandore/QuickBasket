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
    /* Root Layout */
    .main {
        padding: 2rem;
        background-color: #0A0A1F;
    }

    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px 20px;
        color: #F0F8FF;
        background-color: #1B1B3A;
        border: 1px solid #00FFFF;
        transition: border-color 0.3s ease-in-out;
    }

    .stTextInput > div > div > input:hover {
        border-color: #FF00FF;
    }

    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #F0F8FF;
        transition: transform 0.2s ease;
    }

    .chat-message:hover {
        transform: scale(1.02);
    }

    .chat-message.user {
        background-color: #151538;
        border-left: 5px solid #00FFFF;
    }

    .chat-message.assistant {
        background-color: #1F1F4D;
        border-left: 5px solid #FF00FF;
    }

    .chat-message .content {
        margin-top: 0.5rem;
        color: #F0F8FF;
        animation: fadeInUp 1s ease-in-out;
    }

    .chat-message strong {
        color: #00FFC6;
    }

    /* Sidebar Fix for better readability */
    .sidebar .sidebar-content {
        background-color: #1A1A2E !important;
        color: #FFFFFF !important;
    }

    /* Sidebar text elements */
    .sidebar .css-1d391kg, 
    .sidebar .css-1cpxqw2, 
    .sidebar .css-1v0mbdj, 
    .sidebar .css-1x8cf1d {
        color: #FFFFFF !important;
    }

    /* Sidebar links or buttons */
    .sidebar .css-1aumxhk {
        background-color: #2B2B4D !important;
        color: #00FFC6 !important;
        border-radius: 10px;
    }

    .sidebar .css-1aumxhk:hover {
        background-color: #FF00FF !important;
        color: #FFFFFF !important;
        transform: scale(1.05);
    }

    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        background-color: #00FFC6;
        color: #0A0A1F;
        border: none;
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #FF00FF;
        color: white;
        transform: scale(1.05);
    }

    .title-text {
        color: #FF00FF;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        animation: floatText 5s ease-in-out infinite;
    }

    .subtitle-text {
        color: #CCCCFF;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: typing 4s steps(40, end), blink-caret 0.75s step-end infinite;
        white-space: nowrap;
        overflow: hidden;
        border-right: 3px solid #ADD8E6;
    }

    .stMarkdown {
        color: #F0F8FF;
    }

    .stMarkdown p {
        color: #F0F8FF;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #00FFC6;
    }

    .stMarkdown strong {
        color: #FF00FF;
    }

    .stApp {
        background-color: #1A1A3D;
        color: #FFFFFF;
    }

    .stTextInput > div > div > input::placeholder {
        color: #8080FF;
    }

    .stMarkdown code {
        background-color: #1B1B3A;
        color: #00FFC6;
    }

    /* Text animation: fadeIn */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 20%, 0);
        }
        to {
            opacity: 1;
            transform: none;
        }
    }

    /* Floating Title Animation */
    @keyframes floatText {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }

    /* Typing animation */
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #00FFC6; }
    }

    /* Apply Dark Green to the specific list items */
    .stMarkdown ul li {
        color: #006400; /* Dark Green */
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

# Main application
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