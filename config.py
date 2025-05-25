import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
NIKE_API_KEY = os.getenv('NIKE_API_KEY', 'demo_nike_key')
ADIDAS_API_KEY = os.getenv('ADIDAS_API_KEY', 'demo_adidas_key')
PUMA_API_KEY = os.getenv('PUMA_API_KEY', 'demo_puma_key')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'demo_maps_key')
GOOGLE_PAY_MERCHANT_ID = os.getenv('GOOGLE_PAY_MERCHANT_ID', 'demo_merchant_id')

# API Base URLs
NIKE_BASE_URL = "https://api.nike.com/product/v1"
ADIDAS_BASE_URL = "https://api.adidas.com/products/v1"
PUMA_BASE_URL = "https://api.puma.com/catalog/v1"
GOOGLE_MAPS_BASE_URL = "https://maps.googleapis.com/maps/api"
GOOGLE_PAY_BASE_URL = "https://pay.google.com/gp/v1"

# App Settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
MOCK_API_RESPONSES = os.getenv('MOCK_API_RESPONSES', 'True').lower() == 'true'  # Use mock data for development

# Cache Settings
CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '3600'))  # Default 1 hour
MAX_CACHE_ITEMS = int(os.getenv('MAX_CACHE_ITEMS', '1000'))

# API Rate Limits
RATE_LIMIT_CALLS = int(os.getenv('RATE_LIMIT_CALLS', '100'))
RATE_LIMIT_PERIOD = int(os.getenv('RATE_LIMIT_PERIOD', '3600'))  # Period in seconds 