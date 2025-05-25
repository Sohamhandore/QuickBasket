import requests
import json
from typing import Dict, List, Optional
import os
from datetime import datetime
from config import *

# Mock data for development
MOCK_DATA = {
    "nike": {
        "products": [
            {
                "id": "NK12345",
                "name": "Nike Air Max",
                "price": 129.99,
                "category": "running",
                "sizes": [7, 8, 9, 10, 11],
                "colors": ["black", "white", "red"],
                "in_stock": True
            },
            {
                "id": "NK67890",
                "name": "Nike React",
                "price": 149.99,
                "category": "running",
                "sizes": [8, 9, 10],
                "colors": ["blue", "gray"],
                "in_stock": True
            }
        ]
    },
    "adidas": {
        "products": [
            {
                "id": "AD12345",
                "name": "Adidas Ultraboost",
                "price": 179.99,
                "category": "running",
                "sizes": [7, 8, 9, 10, 11],
                "colors": ["black", "white", "blue"],
                "in_stock": True
            }
        ]
    },
    "puma": {
        "products": [
            {
                "id": "PM12345",
                "name": "Puma RS-X",
                "price": 109.99,
                "category": "lifestyle",
                "sizes": [8, 9, 10],
                "colors": ["white", "black"],
                "in_stock": True
            }
        ]
    }
}

# API Configuration
class APIConfig:
    def __init__(self):
        self.NIKE_API_KEY = NIKE_API_KEY
        self.ADIDAS_API_KEY = ADIDAS_API_KEY
        self.PUMA_API_KEY = PUMA_API_KEY
        self.GOOGLE_MAPS_API_KEY = GOOGLE_MAPS_API_KEY
        self.GOOGLE_PAY_MERCHANT_ID = GOOGLE_PAY_MERCHANT_ID

        self.NIKE_BASE_URL = NIKE_BASE_URL
        self.ADIDAS_BASE_URL = ADIDAS_BASE_URL
        self.PUMA_BASE_URL = PUMA_BASE_URL
        self.GOOGLE_MAPS_BASE_URL = GOOGLE_MAPS_BASE_URL
        self.GOOGLE_PAY_BASE_URL = GOOGLE_PAY_BASE_URL

# Nike API Integration
class NikeAPI:
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.NIKE_API_KEY}",
            "Content-Type": "application/json"
        }

    def get_products(self, category: str = None, limit: int = 10) -> List[Dict]:
        """Fetch Nike products"""
        if MOCK_API_RESPONSES:
            products = MOCK_DATA["nike"]["products"]
            if category:
                products = [p for p in products if p["category"] == category]
            return {"products": products[:limit]}
            
        endpoint = f"{self.config.NIKE_BASE_URL}/products"
        params = {"limit": limit}
        if category:
            params["category"] = category
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()

    def get_product_details(self, product_id: str) -> Dict:
        """Get detailed information about a specific Nike product"""
        endpoint = f"{self.config.NIKE_BASE_URL}/products/{product_id}"
        response = requests.get(endpoint, headers=self.headers)
        return response.json()

    def check_availability(self, product_id: str, size: str) -> Dict:
        """Check if a specific Nike product is available in the given size"""
        endpoint = f"{self.config.NIKE_BASE_URL}/products/{product_id}/availability"
        params = {"size": size}
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()

# Adidas API Integration
class AdidasAPI:
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.ADIDAS_API_KEY}",
            "Content-Type": "application/json"
        }

    def get_products(self, category: str = None, limit: int = 10) -> List[Dict]:
        """Fetch Adidas products"""
        if MOCK_API_RESPONSES:
            products = MOCK_DATA["adidas"]["products"]
            if category:
                products = [p for p in products if p["category"] == category]
            return {"products": products[:limit]}
            
        endpoint = f"{self.config.ADIDAS_BASE_URL}/products"
        params = {"limit": limit}
        if category:
            params["category"] = category
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()

    def get_product_details(self, product_id: str) -> Dict:
        """Get detailed information about a specific Adidas product"""
        endpoint = f"{self.config.ADIDAS_BASE_URL}/products/{product_id}"
        response = requests.get(endpoint, headers=self.headers)
        return response.json()

    def check_stock(self, product_id: str, size: str) -> Dict:
        """Check stock availability for an Adidas product"""
        endpoint = f"{self.config.ADIDAS_BASE_URL}/products/{product_id}/stock"
        params = {"size": size}
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()

# Puma API Integration
class PumaAPI:
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.PUMA_API_KEY}",
            "Content-Type": "application/json"
        }

    def get_products(self, category: str = None, limit: int = 10) -> List[Dict]:
        """Fetch Puma products"""
        if MOCK_API_RESPONSES:
            products = MOCK_DATA["puma"]["products"]
            if category:
                products = [p for p in products if p["category"] == category]
            return {"products": products[:limit]}
            
        endpoint = f"{self.config.PUMA_BASE_URL}/products"
        params = {"limit": limit}
        if category:
            params["category"] = category
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()

    def get_product_details(self, product_id: str) -> Dict:
        """Get detailed information about a specific Puma product"""
        endpoint = f"{self.config.PUMA_BASE_URL}/products/{product_id}"
        response = requests.get(endpoint, headers=self.headers)
        return response.json()

    def check_availability(self, product_id: str, size: str) -> Dict:
        """Check product availability in the given size"""
        endpoint = f"{self.config.PUMA_BASE_URL}/products/{product_id}/availability"
        params = {"size": size}
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()

# Google Maps Integration
class GoogleMapsAPI:
    def __init__(self, config: APIConfig):
        self.config = config
        self.api_key = config.GOOGLE_MAPS_API_KEY

    def find_nearby_stores(self, latitude: float, longitude: float, radius: int = 5000) -> List[Dict]:
        """Find nearby Quick Basket stores"""
        endpoint = f"{self.config.GOOGLE_MAPS_BASE_URL}/place/nearbysearch/json"
        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "type": "store",
            "keyword": "Quick Basket",
            "key": self.api_key
        }
        response = requests.get(endpoint, params=params)
        return response.json()

    def get_store_details(self, place_id: str) -> Dict:
        """Get detailed information about a specific store"""
        endpoint = f"{self.config.GOOGLE_MAPS_BASE_URL}/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,formatted_address,opening_hours,formatted_phone_number",
            "key": self.api_key
        }
        response = requests.get(endpoint, params=params)
        return response.json()

    def get_directions(self, origin: str, destination: str) -> Dict:
        """Get directions to a store"""
        endpoint = f"{self.config.GOOGLE_MAPS_BASE_URL}/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "key": self.api_key
        }
        response = requests.get(endpoint, params=params)
        return response.json()

# Google Pay Integration
class GooglePayAPI:
    def __init__(self, config: APIConfig):
        self.config = config
        self.merchant_id = config.GOOGLE_PAY_MERCHANT_ID
        self.base_url = config.GOOGLE_PAY_BASE_URL

    def create_payment_token(self, amount: float, currency: str = "USD") -> Dict:
        """Create a payment token for Google Pay"""
        endpoint = f"{self.base_url}/paymentData"
        payload = {
            "merchantInfo": {
                "merchantId": self.merchant_id,
                "merchantName": "Quick Basket"
            },
            "allowedPaymentMethods": ["CARD", "TOKENIZED_CARD"],
            "transactionInfo": {
                "totalPrice": str(amount),
                "totalPriceStatus": "FINAL",
                "currencyCode": currency
            }
        }
        response = requests.post(endpoint, json=payload)
        return response.json()

    def process_payment(self, payment_token: str, amount: float) -> Dict:
        """Process a payment using Google Pay"""
        endpoint = f"{self.base_url}/processPayment"
        payload = {
            "merchantId": self.merchant_id,
            "paymentToken": payment_token,
            "amount": amount
        }
        response = requests.post(endpoint, json=payload)
        return response.json()

# Main API Manager
class APIManager:
    def __init__(self):
        self.config = APIConfig()
        self.nike = NikeAPI(self.config)
        self.adidas = AdidasAPI(self.config)
        self.puma = PumaAPI(self.config)
        self.google_maps = GoogleMapsAPI(self.config)
        self.google_pay = GooglePayAPI(self.config)

    def get_all_products(self, category: str = None, limit: int = 10) -> Dict[str, List[Dict]]:
        """Fetch products from all brands"""
        if MOCK_API_RESPONSES:
            results = {}
            for brand in ["nike", "adidas", "puma"]:
                products = MOCK_DATA[brand]["products"]
                if category:
                    products = [p for p in products if p["category"] == category]
                results[brand] = {"products": products[:limit]}
            return results
            
        return {
            "nike": self.nike.get_products(category, limit),
            "adidas": self.adidas.get_products(category, limit),
            "puma": self.puma.get_products(category, limit)
        }

    def check_product_availability(self, brand: str, product_id: str, size: str) -> Dict:
        """Check product availability across brands"""
        if brand.lower() == "nike":
            return self.nike.check_availability(product_id, size)
        elif brand.lower() == "adidas":
            return self.adidas.check_stock(product_id, size)
        elif brand.lower() == "puma":
            return self.puma.check_availability(product_id, size)
        else:
            raise ValueError("Unsupported brand")

# Error Handling
class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# Usage Example:
if __name__ == "__main__":
    # Initialize API manager
    api_manager = APIManager()
    
    try:
        # Get products from all brands
        products = api_manager.get_all_products(category="running", limit=5)
        print("Products retrieved successfully")
        
        # Find nearby stores
        stores = api_manager.google_maps.find_nearby_stores(
            latitude=40.7128,
            longitude=-74.0060
        )
        print("Found nearby stores")
        
        # Create a payment token
        payment = api_manager.google_pay.create_payment_token(amount=99.99)
        print("Payment token created")
        
    except APIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
