import streamlit as st
import pandas as pd
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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""
    if 'product_database' not in st.session_state:
        # Create a mock product database
        st.session_state.product_database = {
            "nike": {
                "Air Max": {"price": 120, "sizes": [7, 8, 9, 10, 11], "colors": ["black", "white", "red"], "in_stock": True},
                "React": {"price": 130, "sizes": [8, 9, 10], "colors": ["blue", "gray"], "in_stock": True},
                "Dunk Low": {"price": 100, "sizes": [7, 8, 9], "colors": ["green", "yellow"], "in_stock": False}
            },
            "adidas": {
                "Ultraboost": {"price": 180, "sizes": [7, 8, 9, 10, 11, 12], "colors": ["black", "white", "blue"], "in_stock": True},
                "Stan Smith": {"price": 80, "sizes": [8, 9, 10, 11], "colors": ["white", "green"], "in_stock": True},
                "Gazelle": {"price": 90, "sizes": [7, 8, 9], "colors": ["blue", "red", "black"], "in_stock": True}
            },
            "puma": {
                "RS-X": {"price": 110, "sizes": [8, 9, 10, 11], "colors": ["white", "black", "blue"], "in_stock": True},
                "Suede": {"price": 70, "sizes": [7, 8, 9, 10], "colors": ["black", "blue", "red"], "in_stock": True}
            }
        }
    if 'order_database' not in st.session_state:
        # Create a mock order database
        st.session_state.order_database = {
            "ORD12345": {
                "date": "2023-11-05",
                "items": ["Nike Air Max - Black, Size 10", "Adidas Stan Smith - White, Size 9"],
                "total": 200,
                "status": "Delivered",
                "delivery_date": "2023-11-10",
                "address": "123 Main St, Anytown, USA"
            },
            "ORD67890": {
                "date": "2023-11-15",
                "items": ["Puma RS-X - Blue, Size 8"],
                "total": 110,
                "status": "Shipped",
                "delivery_date": "2023-11-20",
                "address": "456 Elm St, Somewhere, USA"
            },
            "ORD54321": {
                "date": "2023-11-18",
                "items": ["Nike React - Gray, Size 9", "Adidas Ultraboost - Black, Size 10"],
                "total": 310,
                "status": "Processing",
                "delivery_date": "Expected 2023-11-25",
                "address": "789 Oak St, Nowhere, USA"
            }
        }
    if 'store_locations' not in st.session_state:
        # Create mock store locations
        st.session_state.store_locations = [
            {
                "name": "Quick Basket City Center",
                "address": "123 Main Street, Downtown",
                "hours": "9 AM - 9 PM (Mon-Sat), 10 AM - 6 PM (Sun)",
                "phone": "555-123-4567",
                "features": ["Nike Shop-in-shop", "Shoe fitting service", "Click & Collect"]
            },
            {
                "name": "Quick Basket Mall Store",
                "address": "456 Market Square, Metro Mall",
                "hours": "10 AM - 10 PM (Mon-Sun)",
                "phone": "555-765-4321",
                "features": ["Adidas Shop-in-shop", "Running analysis", "Personal shopping"]
            },
            {
                "name": "Quick Basket Outlet",
                "address": "789 Shopping Plaza, near Central Station",
                "hours": "9 AM - 8 PM (Mon-Sat), Closed on Sun",
                "phone": "555-987-6543",
                "features": ["Clearance items", "Bulk purchase discounts", "Large parking"]
            }
        ]

# Load training data
def load_data():
    try:
        df = pd.read_csv('mock_intents.csv')
        return df
    except Exception as e:
        # If file not found, create a sample one
        try:
            df = pd.DataFrame({
                'User_Query': [
                    'Where is my order?',
                    'I want to return my shoes',
                    'Do you have Nike shoes?',
                    'Where is your store located?',
                    'Hello',
                    'What is your return policy?',
                    'I need help finding a product',
                    'Can I exchange my order?',
                    'What are your store hours?',
                    'Do you ship internationally?',
                    'What sizes do you carry?',
                    'Do you have size 10?',
                    'Are there any discounts?',
                    'Do you have any promotions?',
                    'How much is shipping?',
                    'When will my order arrive?',
                    'Do you accept PayPal?',
                    'What payment methods do you accept?',
                    'Can I use Apple Pay?',
                    'Do you have a sale going on?',
                    'What\'s your biggest discount?',
                    'How do I measure my shoe size?',
                    'How long does shipping take?',
                    'Do you have free shipping?',
                    'What\'s your cheapest shipping option?',
                    'What\'s your opinion on politics?',
                    'Tell me a joke',
                    'What do you think about the current president?',
                    'Can you help me with my homework?',
                    'Hey Siri',
                    'How can I invest in stocks?',
                    'How do I fix my sink?',
                    'What\'s the meaning of life?',
                    'Tell me something inappropriate',
                    'Adult content query',
                    'How can I cheat on my test?',
                    'Tell me how to hack someone\'s account',
                    'What do you know about illegal activities?',
                    'giberish text kdkjf',
                    'asdf',
                    '?????',
                    'blah blah',
                    'hmm',
                    'I\'m looking for nikee shoes',
                    'Do you have addidas',
                    'Gimme some jordanns',
                    'I want to retrun my order',
                    'Help with shiping',
                    'payment opshuns',
                    'Need a refudn'
                ],
                'Intent': [
                    'Order Tracking',
                    'Return & Refund Policy',
                    'Product Availability',
                    'Store Location/Hours',
                    'General Greetings',
                    'Return Policy',
                    'Unknown/Other',
                    'Return & Refund Policy',
                    'Store Location/Hours',
                    'Shipping Information',
                    'Size Inquiry',
                    'Size Inquiry',
                    'Promotions & Discounts',
                    'Promotions & Discounts',
                    'Shipping Information',
                    'Order Tracking',
                    'Payment Options',
                    'Payment Options',
                    'Payment Options',
                    'Promotions & Discounts',
                    'Promotions & Discounts',
                    'Size Inquiry',
                    'Shipping Information',
                    'Shipping Information',
                    'Shipping Information',
                    'Out_Of_Scope',
                    'Out_Of_Scope',
                    'Out_Of_Scope',
                    'Out_Of_Scope',
                    'Misunderstood',
                    'Out_Of_Scope',
                    'Out_Of_Scope',
                    'Out_Of_Scope',
                    'Inappropriate',
                    'Inappropriate',
                    'Inappropriate',
                    'Inappropriate',
                    'Inappropriate',
                    'Misunderstood',
                    'Misunderstood',
                    'Misunderstood',
                    'Misunderstood',
                    'Misunderstood',
                    'Product Availability',  # Typo for Nike
                    'Product Availability',  # Typo for Adidas
                    'Product Availability',  # Typo for Jordan
                    'Return & Refund Policy',  # Typo for return
                    'Shipping Information',   # Typo for shipping
                    'Payment Options',        # Typo for options
                    'Return & Refund Policy'  # Typo for refund
                ]
            })
            df.to_csv('mock_intents.csv', index=False)
            return df
        except Exception as e:
            st.error(f"Error creating sample data: {str(e)}")
            return None

# Train model
def train_model(df):
    try:
        texts = df['User_Query'].values
        labels = df['Intent'].values
        
        # Create a TF-IDF vectorizer with improved features
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),     # Unigrams, bigrams, and trigrams
            analyzer='word',
            min_df=2,               # Minimum document frequency
            max_df=0.95,            # Maximum document frequency
            sublinear_tf=True,      # Apply sublinear tf scaling
            use_idf=True,           # Use inverse document frequency
            norm='l2'               # L2 normalization
        )
        
        # Create classifier
        classifier = MultinomialNB(alpha=0.1, fit_prior=True)
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier)
        ])
        
        # Train the model
        pipeline.fit(texts, labels)
        
        return pipeline
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Common typos and corrections for shoe brands and terms
TYPO_CORRECTIONS = {
    # Brand typos
    'nike': ['nik', 'nikee', 'niike', 'nkie', 'nke'],
    'adidas': ['adiddas', 'addidas', 'adidass', 'adias', 'addias', 'adidias'],
    'puma': ['pumma', 'puuma', 'pooma', 'puna'],
    'jordan': ['jordon', 'jordn', 'jordans', 'jordann'],
    
    # Model typos
    'air max': ['airmax', 'air maxs', 'airmaks', 'air macks'],
    'ultraboost': ['ultra boost', 'ultra-boost', 'ultrabost', 'ultra bost', 'ulttraboost'],
    'stan smith': ['stan smth', 'stansmith', 'stan smiths', 'stan smyth'],
    
    # Generic terms
    'shoes': ['shoe', 'sheos', 'shoess', 'shose', 'shoez'],
    'sneakers': ['sneaker', 'sneakrs', 'sneekers', 'sneekrs', 'snickers'],
    'return': ['retrun', 'retrn', 'reutrn', 'returnn'],
    'order': ['odrer', 'orde', 'ordr', 'orderr'],
    'shipping': ['shiping', 'shippin', 'shippping', 'shiping'],
    'delivery': ['delivry', 'delevery', 'deliverry', 'delibery'],
    'discount': ['disscount', 'discont', 'disccount', 'discunt'],
    'payment': ['paymet', 'payement', 'paymnt', 'payemtn']
}

# Inappropriate content keywords
INAPPROPRIATE_KEYWORDS = [
    'sex', 'porn', 'gambling', 'drugs', 'illegal', 'hack', 'crack', 'steal', 
    'offensive', 'profanity', 'explicit', 'adult', 'inappropriate', 'xxx'
]

# Out of scope topics
OUT_OF_SCOPE_TOPICS = [
    'politics', 'religion', 'medical', 'healthcare', 'prescription', 'medicine',
    'legal advice', 'attorney', 'lawyer', 'stocks', 'investments', 'counseling',
    'therapy', 'psychology', 'relationship', 'dating', 'mortgage', 'loan',
    'insurance', 'tax', 'accounting', 'unrelated'
]

# Check for typos and correct them
def check_for_typos(text):
    words = text.lower().split()
    corrected = []
    corrections_made = {}
    
    for word in words:
        corrected_word = word
        for correct_term, typos in TYPO_CORRECTIONS.items():
            if word in typos:
                corrected_word = correct_term
                corrections_made[word] = correct_term
                break
        corrected.append(corrected_word)
    
    corrected_text = ' '.join(corrected)
    return corrected_text, corrections_made

# Check for inappropriate content
def check_inappropriate_content(text):
    text_lower = text.lower()
    for keyword in INAPPROPRIATE_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

# Check if question is out of scope
def check_out_of_scope(text):
    text_lower = text.lower()
    for topic in OUT_OF_SCOPE_TOPICS:
        if topic in text_lower:
            return True
    
    # Advanced checks for unrelated questions
    shopping_keywords = ['shoe', 'sneaker', 'order', 'delivery', 'store', 'purchase', 
                        'buy', 'price', 'cost', 'return', 'refund', 'exchange', 
                        'size', 'color', 'brand', 'nike', 'adidas', 'puma', 'product']
    
    # If the message is long and doesn't contain any shopping keywords, it might be out of scope
    if len(text.split()) > 8:  # Reasonably long question
        if not any(keyword in text_lower for keyword in shopping_keywords):
            return True
    
    return False

# Enhanced entity extraction with typo correction
def extract_entities(text):
    # First correct any typos
    corrected_text, corrections = check_for_typos(text)
    
    entities = {
        'brands': [],
        'models': [],
        'sizes': [],
        'colors': [],
        'corrections': corrections  # Store any corrections made
    }
    
    # Brand detection
    brand_patterns = {
        'nike': ['nike'],
        'adidas': ['adidas'],
        'puma': ['puma'],
        'jordan': ['jordan']
    }
    
    # Model detection
    model_patterns = {
        'air max': ['air max', 'airmax'],
        'ultraboost': ['ultraboost', 'ultra boost'],
        'stan smith': ['stan smith'],
        'dunk': ['dunk'],
        'react': ['react'],
        'rs-x': ['rs-x', 'rs x'],
        'suede': ['suede'],
        'gazelle': ['gazelle']
    }
    
    # Size detection (US sizes 4-15)
    size_pattern = r'\b(size\s+)?([4-9]|1[0-5])(\.5)?\b'
    size_matches = re.findall(size_pattern, corrected_text.lower())
    if size_matches:
        for match in size_matches:
            if match[1]:  # Size number
                size = match[1]
                if match[2]:  # Half size
                    size += match[2]
                entities['sizes'].append(size)
    
    # Color detection
    color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'gray', 'grey']
    
    # Check for colors
    for color in color_list:
        if re.search(r'\b' + color + r'\b', corrected_text.lower()):
            entities['colors'].append(color)
    
    # Check for brands
    for brand, patterns in brand_patterns.items():
        for pattern in patterns:
            if re.search(r'\b' + pattern + r'\b', corrected_text.lower()):
                entities['brands'].append(brand)
                break
    
    # Check for models
    for model, patterns in model_patterns.items():
        for pattern in patterns:
            if re.search(r'\b' + pattern + r'\b', corrected_text.lower()):
                entities['models'].append(model)
                break
    
    # Remove duplicates
    for key in entities:
        if key != 'corrections':
            entities[key] = list(set(entities[key]))
    
    return entities

# Predict intent from user input
def get_intent(text, classifier):
    try:
        # Check for inappropriate content
        if check_inappropriate_content(text):
            return "Inappropriate", 0.95, {"original_text": text}
        
        # Check for out-of-scope questions
        if check_out_of_scope(text):
            return "Out_Of_Scope", 0.9, {"original_text": text}
        
        # Extract entities with typo correction
        entities = extract_entities(text)
        
        # If corrections were made, use the corrected text for intent classification
        corrected_text = text
        if entities['corrections']:
            # Replace each typo with its correction in the original text
            corrected_text = text
            for typo, correction in entities['corrections'].items():
                corrected_text = re.sub(r'\b' + re.escape(typo) + r'\b', correction, corrected_text, flags=re.IGNORECASE)
        else:
            corrected_text = text
        
        # Check for policy-related keywords
        policy_keywords = ['policy', 'rules', 'terms', 'conditions']
        if any(keyword in corrected_text.lower() for keyword in policy_keywords):
            if 'refund' in corrected_text.lower() or 'return' in corrected_text.lower():
                return "Return Policy", 0.9, entities
            return "Unknown/Other", 0.8, entities
            
        # Check for refund-related keywords
        refund_keywords = ['refund', 'return', 'money back', 'cancel', 'exchange']
        if any(keyword in corrected_text.lower() for keyword in refund_keywords):
            if 'policy' in corrected_text.lower():
                return "Return Policy", 0.9, entities
            return "Return & Refund Policy", 0.9, entities
            
        # Check for product availability
        availability_keywords = ['in stock', 'available', 'have', 'sell', 'carry']
        product_keywords = ['nike', 'adidas', 'puma', 'shoes', 'sneakers', 'air max', 'ultraboost']
        
        # Better product availability detection
        if any(keyword in corrected_text.lower() for keyword in availability_keywords) or entities['brands'] or entities['models']:
            if any(product in corrected_text.lower() for product in product_keywords) or entities['brands'] or entities['models']:
                return "Product Availability", 0.9, entities
            
        # Check for order-related keywords
        order_keywords = ['order', 'tracking', 'delivery', 'package', 'shipment']
        if any(keyword in corrected_text.lower() for keyword in order_keywords):
            return "Order Tracking", 0.9, entities
        
        # Check for store-related keywords
        store_keywords = ['store', 'location', 'address', 'hours', 'open']
        if any(keyword in corrected_text.lower() for keyword in store_keywords):
            return "Store Location/Hours", 0.9, entities
            
        # Check for greeting keywords
        greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good afternoon']
        if any(corrected_text.lower().startswith(keyword) for keyword in greeting_keywords):
            return "General Greetings", 0.9, entities
            
        # Check for size-related keywords
        size_keywords = ['size', 'measurement', 'fit', 'sizing', 'large', 'small', 'medium']
        if any(keyword in corrected_text.lower() for keyword in size_keywords):
            return "Size Inquiry", 0.9, entities
            
        # Check for promotion-related keywords
        promo_keywords = ['discount', 'sale', 'promotion', 'offer', 'deal', 'coupon', 'code', '% off']
        if any(keyword in corrected_text.lower() for keyword in promo_keywords):
            return "Promotions & Discounts", 0.9, entities
            
        # Check for shipping-related keywords
        shipping_keywords = ['shipping', 'delivery', 'ship', 'mail', 'postage', 'delivery time', 'how long']
        if any(keyword in corrected_text.lower() for keyword in shipping_keywords) and 'order' not in corrected_text.lower():
            return "Shipping Information", 0.9, entities
            
        # Check for payment-related keywords
        payment_keywords = ['pay', 'payment', 'credit card', 'debit card', 'paypal', 'apple pay', 'google pay']
        if any(keyword in corrected_text.lower() for keyword in payment_keywords):
            return "Payment Options", 0.9, entities
        
        # Use classifier for other cases
        intent = classifier.predict([corrected_text])[0]
        probs = classifier.predict_proba([corrected_text])[0]
        confidence = probs[classifier.classes_ == intent][0]
        
        if confidence < 0.4:
            # Low confidence might indicate misunderstood query
            if len(text.split()) < 3:  # Very short query
                return "Misunderstood", 0.7, entities
            return "Unknown/Other", confidence, entities
            
        return intent, confidence, entities
    except Exception as e:
        st.error(f"Error predicting intent: {str(e)}")
        return "Misunderstood", 0.5, {}

# Response templates
RESPONSE_TEMPLATES = {
    'Order Tracking': [
        "I can help you track your order. Could you provide your order ID?",
        "To track your order, I'll need your order ID. Could you share that?",
        "Please share your order number and I'll check its status right away.",
        "I'd be happy to help you track your package. What's your order number?",
        "Let me check where your order is. Can you provide the order ID?",
        "For order tracking, I'll need your order number please.",
        "I can locate your shipment with your order ID. Do you have it handy?",
        "To find your delivery status, please provide your order reference number."
    ],
    'Return & Refund Policy': [
        "I can help with your return. Could you please provide your order ID?",
        "I'll help you process your refund. Can you share your order number?",
        "To start the return process, I'll need your order details.",
        "Let's get your return started. What's the order number for the items you want to return?",
        "I can assist with processing your return. Which items would you like to return?",
        "For refund assistance, I'll need your order ID and which products you're returning.",
        "Returns are easy with us. Can you share your order number to begin?",
        "I'll guide you through our return process. First, do you have your order confirmation handy?"
    ],
    'Return Policy': [
        "Our return policy allows returns within 30 days of delivery. Items must be unused and in original packaging.",
        "You can return items within 30 days of delivery for a full refund. The item must be in its original condition.",
        "We offer a 30-day return window with full refunds. The product must be unused and in its original packaging.",
        "Quick Basket accepts returns within 30 days of purchase with receipt and original packaging.",
        "Our policy allows 30 days for returns with a valid receipt. Products should be unworn with tags attached.",
        "We have a customer-friendly 30-day return policy. Items should be in resalable condition with original packaging.",
        "Returns are accepted within a month of purchase if items are in original condition with all tags and packaging.",
        "You have 30 days to return products in their original condition. Online orders need the order confirmation number."
    ],
    'Product Availability': [
        "Let me check our inventory for {product} products. Which specific model are you interested in?",
        "We carry {product} products in our stores. What size are you looking for?",
        "Yes, we have {product} items available. Would you like to know about specific models?",
        "I can confirm we stock {product} items. Are you looking for a particular style or size?",
        "{product} is one of our popular brands. Which collection are you interested in?",
        "We have several {product} options in stock. What particular features are you looking for?",
        "Our {product} inventory is well-stocked. Do you have a specific model or color in mind?",
        "Quick Basket carries the latest {product} lines. Can I help you find a specific model?"
    ],
    'Store Location/Hours': [
        "We have stores in City Center, Metro Mall, and near Central Station. Our stores are open Monday-Saturday, 9 AM to 9 PM.",
        "Our main store is at 123 Main Street, Downtown. All our stores are open Monday-Saturday, 9 AM to 9 PM.",
        "Quick Basket has 3 locations. The City Center store is open 9 AM-9 PM (Mon-Sat) and 10 AM-6 PM (Sun).",
        "You can visit us at Metro Mall (10 AM-10 PM daily) or our City Center location (9 AM-9 PM weekdays).",
        "Our nearest store is at 456 Market Square in Metro Mall, open from 10 AM to 10 PM every day.",
        "Quick Basket Outlet near Central Station is open weekdays 9 AM-8 PM, but closed on Sundays.",
        "We have three convenient locations with varying hours. Which area are you closest to?",
        "Our flagship store in City Center features extended hours including Sunday shopping from 10 AM-6 PM."
    ],
    'General Greetings': [
        "Hello! How can I assist you today with sports footwear and apparel?",
        "Hi there! How may I help you with your shopping needs?",
        "Welcome to Quick Basket! I can help with products, orders, or store info.",
        "Good day! I'm your Quick Basket assistant. What brings you here today?",
        "Hey there! How can I make your shopping experience better today?",
        "Hello and welcome! I'm here to help with any Quick Basket questions you might have.",
        "Hi! I'm your virtual shopping assistant. What can I help you find today?",
        "Greetings! How can I assist with your footwear and sports apparel needs today?"
    ],
    'Unknown/Other': [
        "I can help you with our sports footwear collection, including Nike, Adidas, and Puma. What would you like to know?",
        "I can assist you with product availability, orders, returns, and store information. What do you need help with?",
        "I'm here to help with your sports footwear and apparel needs. What specific information are you looking for?",
        "I specialize in helping with orders, product info, returns, and store locations. How can I assist you?",
        "I'd be happy to help with any questions about our products, shipping, returns or store locations.",
        "Quick Basket offers a wide range of athletic footwear. Can you tell me more about what you're looking for?",
        "I can provide information on our shoes, delivery options, or return policies. What are you interested in?",
        "I'm your Quick Basket assistant. I can check stock, help with orders, or find store information for you."
    ],
    'Size Inquiry': [
        "We carry US sizes 4-15 in most styles. Some premium models may have a more limited size range.",
        "Most of our shoes come in US sizes 4-15, including half sizes. Which size are you looking for?",
        "Our size range typically spans from US 4 to 15. Do you know your size in US measurements?",
        "We offer a comprehensive size range from 4 to 15 US. Would you like information on how to measure your foot?",
        "Our shoes typically come in sizes 4-15 US. Some specialty performance models might have different sizing."
    ],
    'Promotions & Discounts': [
        "We currently have a buy-one-get-one 50% off promotion on selected running shoes.",
        "This week we're offering 25% off all Adidas products with code ADIRUN23.",
        "New members get 15% off their first purchase when they sign up for our loyalty program.",
        "Our seasonal clearance has up to 40% off on last season's styles.",
        "Check our website's promotion section for current deals, including our weekend flash sales."
    ],
    'Shipping Information': [
        "Standard shipping takes 3-5 business days and costs $5.99. Orders over $75 ship free.",
        "We offer express shipping (1-2 days) for $12.99 and standard shipping (3-5 days) for $5.99.",
        "Orders placed before 2 PM usually ship same day. Free shipping on orders over $75.",
        "Domestic shipping takes 3-5 business days. International shipping is available to select countries.",
        "We ship to all 50 states and over 25 countries. Delivery times vary by location."
    ],
    'Payment Options': [
        "We accept all major credit cards, PayPal, Apple Pay, and Google Pay.",
        "You can pay with credit/debit cards, PayPal, or our store gift cards.",
        "We support major credit cards, digital wallets, and payment plans through Affirm.",
        "Payment options include credit cards, PayPal, Shop Pay, and interest-free installments.",
        "We accept Visa, Mastercard, American Express, Discover, and various digital payment methods."
    ],
    'Misunderstood': [
        "I'm not sure I understood that. Could you rephrase your question about our footwear or services?",
        "I didn't quite catch that. Can you ask your question about our products or services differently?",
        "I'm still learning and didn't understand your question. Could you try asking in a different way?",
        "I'm having trouble understanding your request. Could you be more specific about what you need?",
        "I apologize, but I'm not sure what you're asking. Could you clarify what you'd like to know about our products or services?"
    ],
    'Out_Of_Scope': [
        "I'm specialized in helping with Quick Basket's products and services. For questions outside of that scope, you might want to try a general search engine.",
        "That's beyond my expertise as a Quick Basket assistant. I can help with our products, orders, returns, and store information.",
        "I'm focused on helping with Quick Basket shopping needs. I'm not able to assist with that specific request.",
        "I can only provide information related to Quick Basket products and services. Could I help you with something in that area?",
        "As a Quick Basket shopping assistant, I'm not able to help with that. Is there something related to our footwear or apparel I can assist with?"
    ],
    'Inappropriate': [
        "I'm here to help with Quick Basket shopping inquiries. Let's keep our conversation focused on our products and services.",
        "I'd be happy to assist with any product or service questions you have about Quick Basket offerings.",
        "Let's focus on how I can help you with your shopping needs at Quick Basket.",
        "I'm designed to assist with shopping at Quick Basket. What products or services can I help you with today?",
        "My purpose is to help with your Quick Basket shopping experience. What footwear or apparel information can I provide?"
    ],
    'Typo_Correction': [
        "I think you're asking about {corrected_term}. Is that right?",
        "Did you mean {corrected_term}? Let me help you with that.",
        "I understand you're interested in {corrected_term}. Here's what I can tell you.",
        "Assuming you meant {corrected_term}, I can provide the following information.",
        "I believe you're looking for information about {corrected_term}. Let me assist you with that."
    ]
}

# Extract product context from message
def get_product_context(message):
    keywords = ['nike', 'adidas', 'puma', 'shoes', 'sneakers']
    for keyword in keywords:
        if keyword.lower() in message.lower():
            return keyword
    return None

# Find order information using order ID
def get_order_info(order_id):
    if order_id in st.session_state.order_database:
        return st.session_state.order_database[order_id]
    return None

# Extract order ID from message
def extract_order_id(message):
    # Look for patterns like "ORD12345"
    order_pattern = r'ORD\d{5}'
    match = re.search(order_pattern, message, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    
    # Look for 5 digits that could be an order number
    num_pattern = r'\b\d{5}\b'
    match = re.search(num_pattern, message)
    if match:
        return f"ORD{match.group(0)}"
    
    return None

# Generate response based on intent
def generate_response(user_input, classifier):
    # Get intent and entities
    intent, confidence, entities = get_intent(user_input, classifier)
    
    # Extract product context if available
    product = get_product_context(user_input)
    
    # Check for order ID
    order_id = extract_order_id(user_input)
    
    # Get base response
    responses = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES['Unknown/Other'])
    response = np.random.choice(responses)
    
    # Handle typo corrections
    if 'corrections' in entities and entities['corrections'] and intent not in ["Inappropriate", "Out_Of_Scope", "Misunderstood"]:
        # If significant corrections were made, append a typo correction message
        if len(entities['corrections']) > 0:
            corrected_terms = ", ".join([f"'{k}' to '{v}'" for k, v in entities['corrections'].items()])
            typo_msg = np.random.choice(RESPONSE_TEMPLATES['Typo_Correction'])
            typo_msg = typo_msg.format(corrected_term=", ".join(entities['corrections'].values()))
            response = f"{typo_msg} {response}"
    
    # Handle inappropriate content
    if intent == "Inappropriate":
        return np.random.choice(RESPONSE_TEMPLATES['Inappropriate'])
    
    # Handle out of scope questions
    if intent == "Out_Of_Scope":
        return np.random.choice(RESPONSE_TEMPLATES['Out_Of_Scope'])
    
    # Handle misunderstood queries
    if intent == "Misunderstood":
        return np.random.choice(RESPONSE_TEMPLATES['Misunderstood'])
    
    # Handle order tracking with order ID
    if intent == "Order Tracking" and order_id:
        order_info = get_order_info(order_id)
        if order_info:
            response = f"I found your order {order_id}. Status: {order_info['status']}. "
            if order_info['status'] == 'Delivered':
                response += f"It was delivered on {order_info['delivery_date']}."
            else:
                response += f"Estimated delivery: {order_info['delivery_date']}."
        else:
            response = f"I couldn't find order {order_id} in our system. Please verify the order number."
    
    # Handle product availability
    elif intent == "Product Availability":
        if entities['brands'] or entities['models']:
            brands = entities['brands']
            models = entities['models']
            sizes = entities['sizes']
            colors = entities['colors']
            
            # Create detailed response based on entities
            if brands and models:
                brand = brands[0]
                model = models[0]
                
                # Check our database
                product_info = None
                if brand.lower() in st.session_state.product_database:
                    for prod_model in st.session_state.product_database[brand.lower()]:
                        if model.lower() in prod_model.lower():
                            product_info = st.session_state.product_database[brand.lower()][prod_model]
                            model = prod_model
                            break
                
                if product_info:
                    if product_info["in_stock"]:
                        response = f"Yes, we have {brand.capitalize()} {model} shoes in stock. "
                        
                        if sizes:
                            size = sizes[0]
                            if float(size) in product_info["sizes"]:
                                response += f"Size {size} is available. "
                            else:
                                response += f"Size {size} is not in stock, but we have sizes {', '.join(map(str, product_info['sizes']))}. "
                        else:
                            response += f"Available sizes: {', '.join(map(str, product_info['sizes']))}. "
                            
                        if colors:
                            color = colors[0]
                            if color in product_info["colors"]:
                                response += f"And yes, we have them in {color}."
                            else:
                                response += f"We don't have them in {color}, but they come in {', '.join(product_info['colors'])}."
                        
                        response += f" The price is ${product_info['price']}."
                    else:
                        response = f"I'm sorry, {brand.capitalize()} {model} shoes are currently out of stock."
                else:
                    response = f"I'll check if we have {brand.capitalize()} {model} in stock. What size are you looking for?"
            elif brands:
                brand = brands[0]
                if brand.lower() in st.session_state.product_database:
                    models_available = list(st.session_state.product_database[brand.lower()].keys())
                    response = f"We carry {brand.capitalize()} products. Available models: {', '.join(models_available)}. Which one interests you?"
                else:
                    response = f"I'll check if we have {brand.capitalize()} products. Which model are you looking for?"
            elif models:
                model = models[0]
                response = f"I can check if we have {model} shoes. Which brand are you interested in?"
            
        elif product:
            response = response.format(product=product.capitalize())
    
    # Handle store location/hours
    elif intent == "Store Location/Hours":
        store_info = st.session_state.store_locations[0]  # Default to first store
        response = f"Our {store_info['name']} is located at {store_info['address']}. Hours: {store_info['hours']}. Phone: {store_info['phone']}."
    
    return response

def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #2196F3;'>Quick Basket Assistant</h2>
                <p style='color: #666;'>AI That Shops with You</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🤖 Features")
        st.markdown("""
            - 📦 Track your order status
            - 🔄 Get help with returns & refunds
            - 🏪 Find store locations & hours
            - 🛍️ Check product availability
            - 👟 Get information on shoes & apparel
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content
    st.markdown("<h1 class='title-text'>Quick Basket Customer Support</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Your Fast, Friendly, and Frugal shopping assistant is here to help!</p>", unsafe_allow_html=True)
    
    # Model initialization
    if st.session_state.classifier is None:
        with st.spinner("🔄 Setting up the assistant..."):
            df = load_data()
            if df is not None:
                classifier = train_model(df)
                if classifier is not None:
                    st.session_state.classifier = classifier
                    st.success("✅ Ready to assist you!")
                else:
                    st.error("❌ Failed to train the assistant.")
            else:
                st.error("❌ Failed to load training data.")
    
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
    
    # User input field
    with st.container():
        user_input = st.text_input("Type your message:", key="user_input", placeholder="Ask me about products, orders, or stores...")
    
    # Process user input
    if user_input and user_input != st.session_state.last_input:
        # Update session state
        st.session_state.last_input = user_input
        
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Generate response
        if st.session_state.classifier is not None:
            response = generate_response(user_input, st.session_state.classifier)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        
        # Rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    main() 
