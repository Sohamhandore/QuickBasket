import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import re
import os
import time

# Set page config
st.set_page_config(
    page_title="Quick Basket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme settings
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Theme colors
THEME_COLORS = {
    "light": {
        "bg_primary": "#FFF8E7",
        "bg_secondary": "#F5E6D3",
        "text_primary": "#4A4A4A",
        "text_secondary": "#8B7355",
        "accent": "#D4A017",
        "accent_secondary": "#B38B0B",
        "card_bg": "#FFFFFF",
        "success": "#28a745",
        "info": "#17a2b8",
        "warning": "#ffc107",
        "danger": "#dc3545"
    },
    "dark": {
        "bg_primary": "#121212",
        "bg_secondary": "#1E1E1E",
        "text_primary": "#E0E0E0",
        "text_secondary": "#B0B0B0",
        "accent": "#FFC107",
        "accent_secondary": "#FFD54F",
        "card_bg": "#2D2D2D",
        "success": "#4CAF50",
        "info": "#03A9F4",
        "warning": "#FF9800",
        "danger": "#F44336"
    }
}

# Get current theme colors
def get_theme():
    return THEME_COLORS[st.session_state.theme]

# Custom CSS with theme colors
def get_custom_css():
    colors = get_theme()
    # Return only the CSS content, without the <style> tags
    return f"""
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap');

    /* Root Layout */
    .main {{
        padding: 2rem;
        background-color: {colors['bg_primary']};
        font-family: 'Poppins', sans-serif;
        color: {colors['text_primary']};
        transition: all 0.3s ease;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: {colors['text_primary']};
    }}

    .title-text {{
        color: {colors['accent']};
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(212, 160, 23, 0.3);
        animation: floatText 5s ease-in-out infinite;
        font-family: 'Montserrat', sans-serif;
    }}

    .subtitle-text {{
        color: {colors['text_secondary']};
        font-size: 1.4rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
        animation: typing 4s steps(40, end), blink-caret 0.75s step-end infinite;
        white-space: nowrap;
        overflow: hidden;
        border-right: 3px solid {colors['accent']};
        font-family: 'Poppins', sans-serif;
    }}

    /* Input Styling */
    .stTextInput > div > div > input {{
        border-radius: 25px;
        padding: 15px 25px;
        color: {colors['text_primary']};
        background-color: {colors['bg_secondary']};
        border: 2px solid {colors['accent']};
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease-in-out;
    }}

    .stTextInput > div > div > input:hover {{
        border-color: {colors['text_secondary']};
        box-shadow: 0 0 15px rgba(139, 115, 85, 0.3);
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {colors['text_secondary']};
        box-shadow: 0 0 20px rgba(139, 115, 85, 0.4);
    }}

    /* Chat Messages */
    .chat-message {{
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        color: {colors['text_primary']};
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }}

    .chat-message:hover {{
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }}

    .chat-message.user {{
        background: linear-gradient(135deg, {colors['bg_secondary']} 0%, {colors['bg_primary']} 100%);
        border-left: 5px solid {colors['accent']};
    }}

    .chat-message.assistant {{
        background: linear-gradient(135deg, {colors['bg_secondary']} 0%, {colors['bg_primary']} 100%);
        border-left: 5px solid {colors['text_secondary']};
    }}

    .chat-message .content {{
        margin-top: 0.8rem;
        color: {colors['text_primary']};
        animation: fadeInUp 0.5s ease-in-out;
    }}

    .chat-message strong {{
        color: {colors['accent']};
        font-weight: 600;
    }}

    /* Sidebar Styling */
    .sidebar .sidebar-content {{
        background-color: {colors['bg_primary']} !important;
        color: {colors['text_primary']} !important;
        font-family: 'Poppins', sans-serif;
    }}

    .sidebar h2 {{
        color: {colors['accent']};
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }}

    .sidebar p {{
        color: {colors['text_secondary']};
        font-size: 1.1rem;
    }}

    /* Button Styling */
    .stButton > button {{
        border-radius: 25px;
        padding: 0.8rem 1.5rem;
        background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%);
        color: #FFFFFF;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {colors['accent_secondary']} 0%, {colors['accent']} 100%);
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139, 115, 85, 0.3);
    }}

    /* Markdown Styling */
    .stMarkdown {{
        color: {colors['text_primary']};
        font-family: 'Poppins', sans-serif;
        line-height: 1.6;
    }}

    .stMarkdown p {{
        color: {colors['text_primary']};
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }}

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {colors['accent']};
        font-family: 'Montserrat', sans-serif;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}

    .stMarkdown strong {{
        color: {colors['text_secondary']};
        font-weight: 600;
    }}

    .stMarkdown code {{
        background-color: {colors['bg_secondary']};
        color: {colors['accent']};
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Consolas', monospace;
    }}

    /* List Styling */
    .stMarkdown ul li {{
        color: {colors['text_secondary']};
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        list-style-type: none;
        padding-left: 1.5rem;
        position: relative;
    }}

    .stMarkdown ul li:before {{
        content: "•";
        color: {colors['accent']};
        position: absolute;
        left: 0;
        font-size: 1.2rem;
    }}

    /* Product Card Styling */
    .product-card {{
        background-color: {colors['card_bg']};
        border-radius: 15px;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid {colors['bg_secondary']};
    }}
    
    .product-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }}
    
    .product-image {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-bottom: 1px solid {colors['bg_secondary']};
    }}
    
    .product-info {{
        padding: 15px;
    }}
    
    .product-title {{
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        color: {colors['accent']};
        margin-bottom: 5px;
        font-size: 1.2rem;
    }}
    
    .product-price {{
        font-weight: bold;
        color: {colors['text_secondary']};
        margin-bottom: 10px;
        font-size: 1.1rem;
    }}
    
    .product-description {{
        color: {colors['text_primary']};
        margin-bottom: 15px;
        font-size: 0.9rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}
    
    /* Toast Notifications */
    .toast {{
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 25px;
        border-radius: 10px;
        color: white;
        opacity: 0;
        transition: all 0.5s ease;
        z-index: 9999;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        animation: fadeInRight 0.5s forwards, fadeOut 0.5s 2.5s forwards;
    }}
    
    .toast.success {{
        background-color: {colors['success']};
    }}
    
    .toast.error {{
        background-color: {colors['danger']};
    }}
    
    .toast.info {{
        background-color: {colors['info']};
    }}

    /* Cart Badge */
    .cart-badge {{
        position: relative;
        display: inline-block;
    }}
    
    .cart-badge[data-count]:after {{
        content: attr(data-count);
        position: absolute;
        top: -10px;
        right: -10px;
        font-size: 0.75rem;
        background: {colors['accent']};
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    /* Theme Toggle Button */
    .theme-toggle {{
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: {colors['bg_secondary']};
        color: {colors['text_primary']};
        margin: 10px 0;
        transition: all 0.3s ease;
    }}
    
    .theme-toggle:hover {{
        background: {colors['accent']};
        color: white;
    }}
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {{
        .title-text {{
            font-size: 2rem;
        }}
        .subtitle-text {{
            font-size: 1.1rem;
        }}
        .chat-message {{
            padding: 1.2rem;
        }}
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translate3d(0, 20px, 0);
        }}
        to {{
            opacity: 1;
            transform: none;
        }}
    }}

    @keyframes floatText {{
        0%, 100% {{
            transform: translateY(0px);
        }}
        50% {{
            transform: translateY(-10px);
        }}
    }}

    @keyframes typing {{
        from {{ width: 0 }}
        to {{ width: 100% }}
    }}

    @keyframes blink-caret {{
        from, to {{ border-color: transparent }}
        50% {{ border-color: {colors['accent']}; }}
    }}
    
    @keyframes fadeInRight {{
        from {{
            opacity: 0;
            transform: translateX(50px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes fadeOut {{
        from {{
            opacity: 1;
        }}
        to {{
            opacity: 0;
        }}
    }}
    """

# Before any content is displayed, completely reset if needed
if st.sidebar.button("🔄 RESET EVERYTHING", key="complete_reset"):
    # Complete reset of all state
    for key in list(st.session_state.keys()):
        if key != 'theme':  # Keep theme preference
            del st.session_state[key]
    st.rerun()

# Force reset of chat history at the beginning to clear any CSS content
if 'chat_history' in st.session_state:
    # Look for CSS-related content and do a complete reset if found
    for message in st.session_state.chat_history:
        content = message.get('content', '')
        if isinstance(content, str) and (
            '@import' in content or 
            '/* ' in content or 
            ' {' in content or
            '.stButton' in content or
            '@keyframes' in content or
            content.count(';') > 5
        ):
            # Reset the entire chat history
            st.session_state.chat_history = []
            st.session_state.last_input = ""
            break

# Apply custom CSS - ensure this is properly wrapped
st.markdown(f"<style>{get_custom_css()}</style>", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""
    if 'product_database' not in st.session_state:
        # Create a mock product database with images
        st.session_state.product_database = {
            "nike": {
                "Air Max": {
                    "price": 120, 
                    "sizes": [7, 8, 9, 10, 11], 
                    "colors": ["black", "white", "red"], 
                    "in_stock": True,
                    "image": "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/a7854567-0321-4590-9b3f-d89f10e5d69a/air-max-90-shoes-N7Tbw0.png",
                    "description": "Iconic cushioning and retro appeal. The Nike Air Max delivers all-day comfort with a visible Max Air unit."
                },
                "React": {
                    "price": 130, 
                    "sizes": [8, 9, 10], 
                    "colors": ["blue", "gray"], 
                    "in_stock": True,
                    "image": "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/48ffb562-9736-4373-9f5a-73868a3b0d79/react-infinity-run-flyknit-mens-running-shoe-RQ484B.png",
                    "description": "The Nike React features soft, responsive foam for smooth transitions and enhanced comfort on every run."
                },
                "Dunk Low": {
                    "price": 100, 
                    "sizes": [7, 8, 9], 
                    "colors": ["green", "yellow"], 
                    "in_stock": False,
                    "image": "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/344ff3e1-d944-499e-be62-5259a39f1845/dunk-low-shoes-26CGQ7.png",
                    "description": "The Nike Dunk Low stays true to its roots with a padded, low-cut collar and iconic design details."
                }
            },
            "adidas": {
                "Ultraboost": {
                    "price": 180, 
                    "sizes": [7, 8, 9, 10, 11, 12], 
                    "colors": ["black", "white", "blue"], 
                    "in_stock": True,
                    "image": "https://assets.adidas.com/images/w_600,f_auto,q_auto/994ad7862f8b4520a647ad7800c4d61c_9366/Ultraboost_22_Shoes_Black_GZ0127_01_standard.jpg",
                    "description": "Experience epic energy with Adidas Ultraboost, featuring responsive cushioning and a supportive fit."
                },
                "Stan Smith": {
                    "price": 80, 
                    "sizes": [8, 9, 10, 11], 
                    "colors": ["white", "green"], 
                    "in_stock": True,
                    "image": "https://assets.adidas.com/images/w_600,f_auto,q_auto/a81eff5e6bd2435c900fad1500aad828_9366/Stan_Smith_Shoes_White_GV7775_01_standard.jpg",
                    "description": "The Adidas Stan Smith is a timeless tennis shoe with clean lines and minimalist styling."
                },
                "Gazelle": {
                    "price": 90, 
                    "sizes": [7, 8, 9], 
                    "colors": ["blue", "red", "black"], 
                    "in_stock": True,
                    "image": "https://assets.adidas.com/images/w_600,f_auto,q_auto/b3d97ded8430413fab3daa3100f3f5de_9366/Gazelle_Shoes_Blue_BB5478_01_standard.jpg",
                    "description": "The iconic Adidas Gazelle features a suede upper and classic 3-Stripes design."
                }
            },
            "puma": {
                "RS-X": {
                    "price": 110, 
                    "sizes": [8, 9, 10, 11], 
                    "colors": ["white", "black", "blue"], 
                    "in_stock": True,
                    "image": "https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_600,h_600/global/380562/01/sv01/fnd/IND/fmt/png/RS-X-Reinvention-Sneakers",
                    "description": "Puma RS-X features bold design and outstanding cushioning for street-ready style."
                },
                "Suede": {
                    "price": 70, 
                    "sizes": [7, 8, 9, 10], 
                    "colors": ["black", "blue", "red"], 
                    "in_stock": True,
                    "image": "https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_600,h_600/global/374915/01/sv01/fnd/IND/fmt/png/Suede-Classic-XXI-Sneakers",
                    "description": "The Puma Suede is a street style icon with a grippy rubber sole and soft suede upper."
                }
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
    # New session state variables for advanced features
    if 'shopping_cart' not in st.session_state:
        st.session_state.shopping_cart = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "favorite_brands": [],
            "favorite_colors": [],
            "viewed_products": [],
            "preferred_sizes": []
        }
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            "current_topic": None,
            "mentioned_brands": [],
            "mentioned_products": [],
            "mentioned_sizes": [],
            "mentioned_colors": [], 
            "last_question_type": None,
            "searches": []
        }
    if 'promotions' not in st.session_state:
        st.session_state.promotions = [
            {"code": "WELCOME10", "discount": 10, "description": "10% off your first purchase"},
            {"code": "SUMMER20", "discount": 20, "description": "20% off summer collection"},
            {"code": "FREESHIP", "discount": "free_shipping", "description": "Free shipping on orders over $50"},
            {"code": "NIKE15", "discount": 15, "description": "15% off Nike products", "brand": "nike"},
            {"code": "ADIDAS25", "discount": 25, "description": "25% off Adidas products", "brand": "adidas"}
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
    ],
    'Shopping_Cart': [
        "I've added that to your cart. Would you like to continue shopping or view your cart?",
        "Item added successfully! You can say 'show my cart' to view your items or continue adding more.",
        "That's now in your shopping cart. Would you like to see anything else?",
        "Added to your cart. Is there anything else you'd like to add before checking out?",
        "Item has been added to your basket. Would you like to see more items or review your cart?"
    ],
    'Recommendation': [
        "Based on your preferences, I think you might like these products as well.",
        "Here are some other items that customers often buy with this product.",
        "You might also be interested in these similar items from our collection.",
        "Since you're interested in this, you might also like to check out these options.",
        "Customers who viewed this item also frequently purchased these related products."
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
    # Check for shopping cart commands first
    cart_command, cart_params = extract_cart_command(user_input)
    
    if cart_command == "add":
        success, message = add_to_cart(
            cart_params["brand"], 
            cart_params["model"], 
            cart_params["size"], 
            cart_params["color"]
        )
        if success:
            # Add to viewed products for future recommendations
            product_key = f"{cart_params['brand']} {cart_params['model']}"
            if product_key not in st.session_state.user_preferences['viewed_products']:
                st.session_state.user_preferences['viewed_products'].append(product_key)
            return message
        else:
            return f"Sorry, I couldn't add that to your cart. {message}"
    
    elif cart_command == "view":
        return format_cart_response()
    
    elif cart_command == "remove":
        # Find the item in the cart
        for i, item in enumerate(st.session_state.shopping_cart):
            if ((cart_params["brand"] and item["brand"].lower() == cart_params["brand"].lower()) or
                (cart_params["model"] and cart_params["model"].lower() in item["model"].lower())):
                success, message = remove_from_cart(i)
                return message
        return "I couldn't find that item in your cart."
    
    elif cart_command == "clear":
        st.session_state.shopping_cart = []
        return "Your cart has been cleared."
    
    # If not a shopping cart command, proceed with intent classification    
    # Get intent and entities
    intent, confidence, entities = get_intent(user_input, classifier)
    
    # Update user preferences and conversation context
    if 'brands' in entities and entities['brands']:
        update_user_preferences(entities, intent)
        update_conversation_context(entities, intent, user_input)
    
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
                
                # Add to viewed products for future recommendations
                product_key = f"{brand} {model}"
                if product_key not in st.session_state.user_preferences['viewed_products']:
                    st.session_state.user_preferences['viewed_products'].append(product_key)
                
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
                        
                        # Add cart prompt
                        response += f"\n\nWould you like to add {brand.capitalize()} {model} to your cart?"
                        
                        # Add product recommendations
                        recommendations = get_similar_products(brand, model)
                        if recommendations:
                            response += format_recommendation_response(recommendations)
                    else:
                        response = f"I'm sorry, {brand.capitalize()} {model} shoes are currently out of stock."
                        
                        # Suggest alternatives based on preferences
                        recommendations = get_personalized_recommendations()
                        if recommendations:
                            response += "\n\nHere are some alternatives you might like:"
                            response += format_recommendation_response(recommendations)
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
    
    # Add personalized recommendations for general inquiries
    elif intent == "General Greetings" or intent == "Unknown/Other":
        if st.session_state.user_preferences['favorite_brands'] or st.session_state.user_preferences['viewed_products']:
            recommendations = get_personalized_recommendations()
            if recommendations:
                response += "\n\nBased on your interests, you might like these products:"
                response += format_recommendation_response(recommendations)
    
    # If this is the first greeting, offer a promotion
    if intent == "General Greetings" and len(st.session_state.chat_history) < 3:
        welcome_promo = next((p for p in st.session_state.promotions if p["code"] == "WELCOME10"), None)
        if welcome_promo:
            response += f"\n\nAs a new customer, you can use code {welcome_promo['code']} for {welcome_promo['discount']}% off your first purchase!"
    
    return response

# Product recommendation functions
def update_user_preferences(entities, intent):
    """Update user preferences based on their queries and conversation"""
    # Update favorite brands
    if entities['brands']:
        for brand in entities['brands']:
            if brand not in st.session_state.user_preferences['favorite_brands']:
                st.session_state.user_preferences['favorite_brands'].append(brand)
    
    # Update favorite colors
    if entities['colors']:
        for color in entities['colors']:
            if color not in st.session_state.user_preferences['favorite_colors']:
                st.session_state.user_preferences['favorite_colors'].append(color)
    
    # Update preferred sizes
    if entities['sizes']:
        for size in entities['sizes']:
            if size not in st.session_state.user_preferences['preferred_sizes']:
                st.session_state.user_preferences['preferred_sizes'].append(size)

def update_conversation_context(entities, intent, user_input):
    """Track conversation context for more coherent multi-turn dialogues"""
    # Update current topic
    st.session_state.conversation_context['last_question_type'] = intent
    
    # Update brands, products, sizes mentioned in this conversation
    if entities['brands']:
        for brand in entities['brands']:
            if brand not in st.session_state.conversation_context['mentioned_brands']:
                st.session_state.conversation_context['mentioned_brands'].append(brand)
    
    if entities['models']:
        for model in entities['models']:
            if model not in st.session_state.conversation_context['mentioned_products']:
                st.session_state.conversation_context['mentioned_products'].append(model)
    
    if entities['sizes']:
        for size in entities['sizes']:
            if size not in st.session_state.conversation_context['mentioned_sizes']:
                st.session_state.conversation_context['mentioned_sizes'].append(size)
    
    if entities['colors']:
        for color in entities['colors']:
            if color not in st.session_state.conversation_context['mentioned_colors']:
                st.session_state.conversation_context['mentioned_colors'].append(color)

def get_similar_products(brand, model, limit=2):
    """Get similar products to recommend"""
    similar_products = []
    
    # If we know the brand and model
    if brand and brand.lower() in st.session_state.product_database:
        target_price = None
        
        # Find the price of the current product
        for prod_model, details in st.session_state.product_database[brand.lower()].items():
            if model.lower() in prod_model.lower():
                target_price = details["price"]
                break
        
        # Find similar products from the same brand
        for prod_model, details in st.session_state.product_database[brand.lower()].items():
            if model.lower() not in prod_model.lower() and details["in_stock"]:
                similar_products.append({
                    "brand": brand,
                    "model": prod_model,
                    "price": details["price"],
                    "similarity": "same brand"
                })
        
        # If we have a price, also find products with similar price points
        if target_price:
            for other_brand, models in st.session_state.product_database.items():
                if other_brand != brand.lower():
                    for other_model, details in models.items():
                        if details["in_stock"] and abs(details["price"] - target_price) <= 30:
                            similar_products.append({
                                "brand": other_brand.capitalize(),
                                "model": other_model,
                                "price": details["price"],
                                "similarity": "similar price"
                            })
    
    # If we only know the brand
    elif brand and brand.lower() in st.session_state.product_database:
        # Recommend products from this brand
        for prod_model, details in st.session_state.product_database[brand.lower()].items():
            if details["in_stock"]:
                similar_products.append({
                    "brand": brand,
                    "model": prod_model,
                    "price": details["price"],
                    "similarity": "popular model"
                })
    
    # If we don't have enough recommendations yet, add some based on preferences
    if len(similar_products) < limit and st.session_state.user_preferences['favorite_brands']:
        # Add products from favorite brands
        for fav_brand in st.session_state.user_preferences['favorite_brands']:
            if fav_brand.lower() in st.session_state.product_database:
                for prod_model, details in st.session_state.product_database[fav_brand.lower()].items():
                    if details["in_stock"]:
                        product = {
                            "brand": fav_brand.capitalize(),
                            "model": prod_model,
                            "price": details["price"],
                            "similarity": "from favorite brand"
                        }
                        if product not in similar_products:
                            similar_products.append(product)
    
    # Sort by price and limit results
    similar_products = sorted(similar_products, key=lambda x: x["price"])
    return similar_products[:limit]

def get_personalized_recommendations(limit=2):
    """Get personalized recommendations based on user preferences"""
    recommendations = []
    
    # First check favorite brands
    if st.session_state.user_preferences["favorite_brands"]:
        for brand in st.session_state.user_preferences["favorite_brands"]:
            if brand.lower() in st.session_state.product_database:
                for model, details in st.session_state.product_database[brand.lower()].items():
                    if details["in_stock"]:
                        # Check if this matches user's preferred sizes
                        size_match = False
                        if st.session_state.user_preferences["preferred_sizes"]:
                            for size in st.session_state.user_preferences["preferred_sizes"]:
                                if float(size) in details["sizes"]:
                                    size_match = True
                                    break
                        else:
                            size_match = True
                            
                        # Check if this matches user's preferred colors
                        color_match = False
                        if st.session_state.user_preferences["favorite_colors"]:
                            for color in st.session_state.user_preferences["favorite_colors"]:
                                if color in details["colors"]:
                                    color_match = True
                                    break
                        else:
                            color_match = True
                        
                        # Add if it matches preferences
                        if size_match or color_match:
                            recommendations.append({
                                "brand": brand.capitalize(),
                                "model": model,
                                "price": details["price"],
                                "reason": "Based on your preferences"
                            })
    
    # If still not enough recommendations, add some popular products
    if len(recommendations) < limit:
        # Just add some popular products (could be enhanced with real popularity metrics)
        popular_products = [
            {"brand": "nike", "model": "Air Max"},
            {"brand": "adidas", "model": "Ultraboost"},
            {"brand": "puma", "model": "RS-X"}
        ]
        
        for product in popular_products:
            if product["brand"] in st.session_state.product_database:
                if product["model"] in st.session_state.product_database[product["brand"]]:
                    details = st.session_state.product_database[product["brand"]][product["model"]]
                    if details["in_stock"]:
                        recommendations.append({
                            "brand": product["brand"].capitalize(),
                            "model": product["model"],
                            "price": details["price"],
                            "reason": "Popular choice"
                        })
    
    # Remove duplicates and limit results
    unique_recommendations = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)
    
    return unique_recommendations[:limit]

def format_recommendation_response(recommendations):
    """Format recommendations into a readable response"""
    if not recommendations:
        return ""
    
    response = "\n\nYou might also like: "
    for i, rec in enumerate(recommendations):
        if i > 0:
            response += " | "
        response += f"{rec['brand']} {rec['model']} (${rec['price']})"
        if 'reason' in rec:
            response += f" - {rec['reason']}"
        elif 'similarity' in rec:
            response += f" - {rec['similarity']}"
    
    return response

# Shopping cart functions
def add_to_cart(brand, model, size=None, color=None, quantity=1):
    """Add a product to the shopping cart"""
    if brand.lower() not in st.session_state.product_database:
        return False, "Brand not found"
    
    # Find the exact model
    model_found = False
    product_details = None
    actual_model_name = None
    
    for prod_model, details in st.session_state.product_database[brand.lower()].items():
        if model.lower() in prod_model.lower():
            model_found = True
            product_details = details
            actual_model_name = prod_model
            break
    
    if not model_found:
        return False, "Model not found"
    
    # Check if in stock
    if not product_details["in_stock"]:
        return False, "Product is out of stock"
    
    # Validate size if provided
    if size and float(size) not in product_details["sizes"]:
        return False, f"Size {size} not available for this model"
    
    # Validate color if provided
    if color and color not in product_details["colors"]:
        return False, f"{color.capitalize()} color not available for this model"
    
    # Default to first available size/color if not specified
    if not size:
        size = product_details["sizes"][0]
    if not color:
        color = product_details["colors"][0]
    
    # Add to cart
    cart_item = {
        "brand": brand.capitalize(),
        "model": actual_model_name,
        "size": size,
        "color": color,
        "price": product_details["price"],
        "quantity": quantity,
        "item_total": product_details["price"] * quantity
    }
    
    # Check if the item is already in the cart (same brand, model, size, color)
    for i, item in enumerate(st.session_state.shopping_cart):
        if (item["brand"].lower() == brand.lower() and 
            item["model"] == actual_model_name and 
            item["size"] == size and 
            item["color"] == color):
            # Update quantity instead of adding a new item
            st.session_state.shopping_cart[i]["quantity"] += quantity
            st.session_state.shopping_cart[i]["item_total"] = st.session_state.shopping_cart[i]["price"] * st.session_state.shopping_cart[i]["quantity"]
            return True, f"Updated {brand.capitalize()} {actual_model_name} in your cart"
    
    # Add as a new item
    st.session_state.shopping_cart.append(cart_item)
    return True, f"Added {brand.capitalize()} {actual_model_name} to your cart"

def remove_from_cart(index):
    """Remove an item from the shopping cart by index"""
    if index < 0 or index >= len(st.session_state.shopping_cart):
        return False, "Invalid item index"
    
    removed_item = st.session_state.shopping_cart.pop(index)
    return True, f"Removed {removed_item['brand']} {removed_item['model']} from your cart"

def update_cart_quantity(index, new_quantity):
    """Update the quantity of an item in the cart"""
    if index < 0 or index >= len(st.session_state.shopping_cart):
        return False, "Invalid item index"
    
    if new_quantity <= 0:
        return remove_from_cart(index)
    
    st.session_state.shopping_cart[index]["quantity"] = new_quantity
    st.session_state.shopping_cart[index]["item_total"] = st.session_state.shopping_cart[index]["price"] * new_quantity
    return True, f"Updated quantity for {st.session_state.shopping_cart[index]['brand']} {st.session_state.shopping_cart[index]['model']}"

def get_cart_total():
    """Calculate the total price of items in the cart"""
    total = 0
    for item in st.session_state.shopping_cart:
        total += item["item_total"]
    return total

def format_cart_response():
    """Format shopping cart into a readable response"""
    if not st.session_state.shopping_cart:
        return "Your cart is empty."
    
    response = "Your shopping cart:\n"
    for i, item in enumerate(st.session_state.shopping_cart):
        response += f"{i+1}. {item['brand']} {item['model']} - {item['color'].capitalize()}, Size {item['size']}, ${item['price']} x {item['quantity']} = ${item['item_total']}\n"
    
    response += f"\nTotal: ${get_cart_total()}"
    
    # Add promotion code suggestions
    relevant_promos = []
    for promo in st.session_state.promotions:
        # Check if this promotion applies to anything in the cart
        if "brand" in promo:
            for item in st.session_state.shopping_cart:
                if item["brand"].lower() == promo["brand"]:
                    relevant_promos.append(promo)
                    break
        else:
            relevant_promos.append(promo)
    
    if relevant_promos:
        response += "\n\nAvailable promotions:"
        for promo in relevant_promos[:2]:  # Limit to 2 promotions
            discount = f"{promo['discount']}% off" if isinstance(promo['discount'], (int, float)) else promo['discount']
            response += f"\n• Use code {promo['code']} for {discount} - {promo['description']}"
    
    return response

def extract_cart_command(text):
    """Extract shopping cart commands from user input"""
    text_lower = text.lower()
    
    # Check for add to cart commands
    add_patterns = [
        r'add (.*) to (?:my )?cart',
        r'buy (.*)',
        r'purchase (.*)',
        r'get (.*)'
    ]
    
    for pattern in add_patterns:
        match = re.search(pattern, text_lower)
        if match:
            product_desc = match.group(1)
            # Extract brand, model, size, color from product description
            entities = extract_entities(product_desc)
            
            if entities['brands'] and entities['models']:
                return "add", {
                    "brand": entities['brands'][0],
                    "model": entities['models'][0],
                    "size": entities['sizes'][0] if entities['sizes'] else None,
                    "color": entities['colors'][0] if entities['colors'] else None
                }
    
    # Check for view cart commands
    view_patterns = [
        r'view (?:my )?cart',
        r'show (?:my )?cart',
        r'check (?:my )?cart',
        r'what\'s in (?:my )?cart',
        r'display cart'
    ]
    
    for pattern in view_patterns:
        if re.search(pattern, text_lower):
            return "view", {}
    
    # Check for remove from cart commands
    remove_patterns = [
        r'remove (.*) from (?:my )?cart',
        r'delete (.*) from (?:my )?cart'
    ]
    
    for pattern in remove_patterns:
        match = re.search(pattern, text_lower)
        if match:
            product_desc = match.group(1)
            entities = extract_entities(product_desc)
            
            if entities['brands'] or entities['models']:
                return "remove", {
                    "brand": entities['brands'][0] if entities['brands'] else None,
                    "model": entities['models'][0] if entities['models'] else None
                }
    
    # Check for clear cart commands
    clear_patterns = [
        r'clear (?:my )?cart',
        r'empty (?:my )?cart'
    ]
    
    for pattern in clear_patterns:
        if re.search(pattern, text_lower):
            return "clear", {}
    
    return None, {}

# Size guide data
SIZE_GUIDE = {
    "US": [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14, 15],
    "UK": [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13.5, 14.5],
    "EU": [36, 37, 37.5, 38, 38.5, 39, 40, 41, 42, 42.5, 43, 44, 45, 45.5, 46, 47, 47.5, 48, 49, 50, 51],
    "CM": [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 32, 33]
}

# Function to show size guide
def show_size_guide():
    st.markdown("### Shoe Size Conversion Chart")
    
    # Create a DataFrame for the size chart
    size_df = pd.DataFrame(SIZE_GUIDE)
    st.table(size_df)
    
    # Foot measurement guide
    st.markdown("### How to Measure Your Foot")
    
    col1, col2 = st.columns([1, 2])
    
    # Add measuring instructions with an image
    col1.image("https://cdn.shopify.com/s/files/1/0419/1525/files/Men-sizing-guide.jpg?v=1589326527", caption="Foot Measurement Guide")
    
    col2.markdown("""
    1. **Place your foot on a piece of paper**
    2. **Trace around your foot**
    3. **Measure the length from heel to toe**
    4. **Use the chart to find your size**
    
    For the best fit:
    - Measure your feet in the evening (when they're largest)
    - Wear the socks you plan to wear with your shoes
    - Measure both feet and use the larger measurement
    """)
    
    # Interactive size finder
    st.markdown("### Find Your Size")
    
    # Input options
    measurement_type = st.selectbox("Measurement Type", ["Foot Length (cm)", "Current Size"])
    
    if measurement_type == "Foot Length (cm)":
        foot_length = st.slider("Foot Length in cm", min_value=22.0, max_value=33.0, value=26.0, step=0.5)
        
        # Find the closest size
        closest_index = min(range(len(SIZE_GUIDE["CM"])), key=lambda i: abs(SIZE_GUIDE["CM"][i] - foot_length))
        
        # Display results
        st.markdown(f"""
        ### Your Recommended Size:
        - US: **{SIZE_GUIDE["US"][closest_index]}**
        - UK: **{SIZE_GUIDE["UK"][closest_index]}**
        - EU: **{SIZE_GUIDE["EU"][closest_index]}**
        - CM: **{SIZE_GUIDE["CM"][closest_index]}**
        """)
        
    else:
        col1, col2 = st.columns(2)
        current_size_system = col1.selectbox("Current Size System", ["US", "UK", "EU"])
        
        # Dynamically set min/max values based on the selected system
        min_size = min(SIZE_GUIDE[current_size_system])
        max_size = max(SIZE_GUIDE[current_size_system])
        
        current_size = col2.number_input(f"Your Current {current_size_system} Size", 
                                         min_value=float(min_size), 
                                         max_value=float(max_size),
                                         value=float(SIZE_GUIDE[current_size_system][10]),  # Default to median
                                         step=0.5)
        
        # Find the closest size in each system
        closest_index = min(range(len(SIZE_GUIDE[current_size_system])), 
                            key=lambda i: abs(SIZE_GUIDE[current_size_system][i] - current_size))
        
        # Display conversion results
        st.markdown("### Size Conversion Results:")
        for system in ["US", "UK", "EU", "CM"]:
            if system != current_size_system:
                st.markdown(f"{system}: **{SIZE_GUIDE[system][closest_index]}**")

def is_css_content(text):
    """Check if the text appears to be CSS content - extremely thorough version"""
    if not text or not isinstance(text, str):
        return False
        
    # Common CSS patterns that strongly indicate CSS content
    strong_indicators = [
        "@import url(",
        "@keyframes",
        "@media",
        "font-family:",
        "margin:",
        "padding:",
        "display: flex",
        "position: absolute",
        "background-color:",
        "linear-gradient("
    ]
    
    # If text contains any strong indicators, it's likely CSS
    for indicator in strong_indicators:
        if indicator in text:
            return True
    
    # CSS structure patterns
    css_structural_patterns = [
        r'\s*\.\w+\s*{',             # .classname {
        r'\s*#\w+\s*{',              # #id {
        r'\s*\w+\s*{\s*\w+:',        # element { property:
        r'}\s*\.\w+\s*{',            # } .classname {
        r'}\s*#\w+\s*{',             # } #id {
        r'\s*@\w+\s*{',              # @media/keyframes {
        r':[^;{]+;',                 # :pseudo-class
        r'}\s*$'                     # ending with }
    ]
    
    # Check for CSS structure patterns
    for pattern in css_structural_patterns:
        if re.search(pattern, text):
            return True
    
    # Check for typical CSS property-value pairs
    css_property_pattern = r'(\s*[\w-]+\s*:\s*[^;{]+\s*;)'
    property_count = len(re.findall(css_property_pattern, text))
    
    # If we have multiple property-value pairs, it's likely CSS
    if property_count > 3:
        return True
        
    # Count brace pairs and semicolons - high counts indicate CSS
    opening_braces = text.count('{')
    closing_braces = text.count('}')
    semicolons = text.count(';')
    
    # If balanced braces and many semicolons, it's likely CSS
    if opening_braces > 1 and opening_braces == closing_braces and semicolons > 5:
        return True
        
    # If long text with CSS-like patterns or many semicolons, likely CSS
    if len(text) > 200 and (semicolons > 10 or ('{' in text and '}' in text)):
        return True
    
    return False

# Clean chat history of any CSS content
def clean_chat_history():
    """Remove any messages that appear to be CSS code"""
    if 'chat_history' in st.session_state:
        cleaned_history = []
        for message in st.session_state.chat_history:
            if not is_css_content(message['content']):
                cleaned_history.append(message)
        st.session_state.chat_history = cleaned_history

# Call this when the app initializes
if 'css_cleaned' not in st.session_state:
    clean_chat_history()
    st.session_state.css_cleaned = True

# Add this function to handle CSS content in user messages
def handle_css_message(css_text):
    """Handle a message that appears to contain CSS content"""
    # Don't add CSS text to chat history
    response = """
    I noticed your message contains CSS code. Since this app already has styling applied, 
    I won't display the CSS as text to avoid conflicts. 
    
    If you're trying to customize the styling, please note that in Streamlit:
    1. CSS is applied using st.markdown with unsafe_allow_html=True
    2. The CSS should be wrapped in <style> tags
    3. You can use the theme toggle feature in the sidebar instead
    
    If you're experiencing CSS issues, try the 'RESET EVERYTHING' button in the sidebar.
    """
    
    # Add only the response to chat history
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
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
        
        # Theme Toggle
        theme_emoji = "🌙" if st.session_state.theme == "light" else "☀️"
        theme_text = "Dark Mode" if st.session_state.theme == "light" else "Light Mode"
        
        if st.button(f"{theme_emoji} {theme_text}", key="theme_toggle"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🤖 Features")
        st.markdown("""
            - 📦 Track your order status
            - 🔄 Get help with returns & refunds
            - 🏪 Find store locations & hours
            - 🛍️ Check product availability
            - 👟 Get information on shoes & apparel
            - 🛒 Add products to your shopping cart
            - 💡 Get personalized recommendations
            - 📏 Interactive size guide
        """)
        
        st.markdown("---")
        
        # Show shopping cart in sidebar
        if st.session_state.shopping_cart:
            cart_count = sum(item['quantity'] for item in st.session_state.shopping_cart)
            st.markdown(f"### 🛒 Your Shopping Cart ({cart_count} items)")
            
            total = 0
            for item in st.session_state.shopping_cart:
                st.markdown("---")
                
                # Display product info
                st.markdown(f"**{item['brand']} {item['model']}**")
                st.markdown(f"Size: {item['size']} | Color: {item['color'].capitalize()}")
                st.markdown(f"Quantity: {item['quantity']} | Price: ${item['item_total']}")
                
                # Get item index
                item_index = st.session_state.shopping_cart.index(item)
                
                # Simple quantity controls without nested columns
                if st.button("-", key=f"dec_{item_index}"):
                    if item['quantity'] > 1:
                        st.session_state.shopping_cart[item_index]["quantity"] -= 1
                        st.session_state.shopping_cart[item_index]["item_total"] = st.session_state.shopping_cart[item_index]["price"] * st.session_state.shopping_cart[item_index]["quantity"]
                        st.rerun()
                    else:
                        # Remove if quantity would be 0
                        st.session_state.shopping_cart.pop(item_index)
                        st.rerun()
                
                if st.button("+", key=f"inc_{item_index}"):
                    st.session_state.shopping_cart[item_index]["quantity"] += 1
                    st.session_state.shopping_cart[item_index]["item_total"] = st.session_state.shopping_cart[item_index]["price"] * st.session_state.shopping_cart[item_index]["quantity"]
                    st.rerun()
                
                # Remove button
                if st.button("🗑️ Remove", key=f"remove_{item_index}"):
                    st.session_state.shopping_cart.pop(item_index)
                    st.rerun()
                
                total += item['item_total']
            
            st.markdown("---")
            st.markdown(f"**Total: ${total}**")
            
            # Checkout and Clear buttons
            if st.button("🛒 Checkout"):
                st.session_state.shopping_cart = []
                st.success("Order placed successfully! Your cart has been cleared.")
                st.rerun()
            
            if st.button("🗑️ Clear Cart"):
                st.session_state.shopping_cart = []
                st.rerun()
        
        # Show some sample commands
        st.markdown("### 💬 Sample Commands")
        st.markdown("""
            - "Do you have Nike Air Max in size 10?"
            - "Add Adidas Ultraboost to cart"
            - "Show my cart"
            - "What's the return policy?"
            - "Where is your store located?"
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content
    st.markdown("<h1 class='title-text'>Quick Basket Customer Support</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Your Fast, Friendly, and Frugal shopping assistant is here to help!</p>", unsafe_allow_html=True)
    
    # Emergency Reset Button - placed at the top for easy access
    col1, col2, col3 = st.columns([1, 4, 3])
    if col1.button("🔄 RESET", key="emergency_reset", help="Reset chat history"):
        # Complete reset of the chat
        st.session_state.chat_history = []
        st.session_state.last_input = ""
        st.session_state.css_cleaned = True
        st.success("Chat completely reset!")
        st.rerun()
    
    # Emergency note if CSS content is detected in chat history - REMOVED
    
    # Tabs for different sections
    chat_tab, catalog_tab, size_guide_tab, orders_tab = st.tabs(["💬 Chat", "👟 Product Catalog", "📏 Size Guide", "📦 Your Orders"])
    
    with chat_tab:
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
            # Check if input appears to be CSS code
            if is_css_content(user_input):
                st.session_state.last_input = user_input  # Mark as processed
                handle_css_message(user_input)
                st.rerun()
            else:
                # Update session state
                st.session_state.last_input = user_input
                
                # Add user message to chat history
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
                # Generate response
                if st.session_state.classifier is not None:
                    response = generate_response(user_input, st.session_state.classifier)
                    
                    # Check if response contains CSS content
                    if is_css_content(response):
                        # Replace with a safe message
                        response = "I generated a response with styling information, but I'll omit it to avoid display issues. Please try asking in a different way."
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                
                # Rerun to update the UI
                st.rerun()
    
    # Product catalog tab
    with catalog_tab:
        st.markdown("### 👟 Browse Our Products")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        brand_filter = col1.selectbox("Brand", ["All"] + list({brand.capitalize() for brand in st.session_state.product_database.keys()}))
        price_range = col2.slider("Price Range", 0, 200, (0, 200))
        show_in_stock_only = col3.checkbox("In Stock Only", True)
        
        # Display products in a grid
        st.markdown("#### Featured Products")
        
        # Get all products that match the filters
        filtered_products = []
        
        for brand, models in st.session_state.product_database.items():
            if brand_filter == "All" or brand.capitalize() == brand_filter:
                for model_name, details in models.items():
                    if price_range[0] <= details["price"] <= price_range[1]:
                        if not show_in_stock_only or details["in_stock"]:
                            filtered_products.append({
                                "brand": brand.capitalize(),
                                "model": model_name,
                                "details": details
                            })
        
        # Display products in rows of 3
        num_products = len(filtered_products)
        rows = (num_products + 2) // 3  # Ceiling division
        
        for row in range(rows):
            cols = st.columns(3)
            for col_idx in range(3):
                product_idx = row * 3 + col_idx
                if product_idx < num_products:
                    product = filtered_products[product_idx]
                    
                    # Product card with HTML
                    cols[col_idx].markdown(f"""
                        <div class="product-card">
                            <img src="{product['details']['image']}" alt="{product['brand']} {product['model']}" class="product-image">
                            <div class="product-info">
                                <div class="product-title">{product['brand']} {product['model']}</div>
                                <div class="product-price">${product['details']['price']}</div>
                                <div class="product-description">{product['details']['description']}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to cart button
                    if product['details']['in_stock']:
                        # Size and color selections
                        size_options = [f"Size {size}" for size in product['details']['sizes']]
                        color_options = [color.capitalize() for color in product['details']['colors']]
                        
                        selected_size = cols[col_idx].selectbox(
                            "Size", 
                            size_options,
                            key=f"size_{product['brand']}_{product['model']}"
                        )
                        
                        selected_color = cols[col_idx].selectbox(
                            "Color", 
                            color_options,
                            key=f"color_{product['brand']}_{product['model']}"
                        )
                        
                        # Extract numeric size value
                        size_value = float(selected_size.replace("Size ", ""))
                        color_value = selected_color.lower()
                        
                        if cols[col_idx].button("Add to Cart", key=f"add_{product['brand']}_{product['model']}"):
                            success, message = add_to_cart(
                                product['brand'],
                                product['model'],
                                size_value,
                                color_value
                            )
                            
                            if success:
                                st.success("Product added to cart!")
                                
                                # Add to viewed products
                                product_key = f"{product['brand']} {product['model']}"
                                if product_key not in st.session_state.user_preferences['viewed_products']:
                                    st.session_state.user_preferences['viewed_products'].append(product_key)
                                
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        cols[col_idx].markdown("<span style='color: red;'>Out of Stock</span>", unsafe_allow_html=True)
    
    # Size guide tab
    with size_guide_tab:
        show_size_guide()
    
    # Orders tab
    with orders_tab:
        st.markdown("### 📦 Your Orders")
        
        # Show sample order history
        if st.session_state.order_database:
            for order_id, order_details in st.session_state.order_database.items():
                with st.expander(f"Order #{order_id} - {order_details['date']}"):
                    st.markdown(f"**Status:** {order_details['status']}")
                    st.markdown(f"**Delivery:** {order_details['delivery_date']}")
                    st.markdown(f"**Ship to:** {order_details['address']}")
                    
                    st.markdown("**Items:**")
                    for item in order_details['items']:
                        st.markdown(f"• {item}")
                    
                    st.markdown(f"**Total:** ${order_details['total']}")
                    
                    # Track or reorder buttons
                    col1, col2 = st.columns(2)
                    if col1.button("Track Shipment", key=f"track_{order_id}"):
                        st.info(f"Tracking information for order {order_id}: {order_details['status']}")
                    
                    if col2.button("Reorder", key=f"reorder_{order_id}"):
                        st.info("This would add all items to cart in a real implementation.")
        else:
            st.info("You don't have any orders yet. Start shopping to see your order history here!")

if __name__ == "__main__":
    main() 
