from api_integrations import APIManager
import json

def test_api_integrations():
    # Initialize API manager
    api_manager = APIManager()
    
    print("Testing API Integrations...")
    
    # Test getting all products
    print("\n1. Testing Get All Products:")
    try:
        products = api_manager.get_all_products(category="running", limit=2)
        print(json.dumps(products, indent=2))
    except Exception as e:
        print(f"Error getting products: {str(e)}")
    
    # Test Google Maps integration
    print("\n2. Testing Google Maps API:")
    try:
        stores = api_manager.google_maps.find_nearby_stores(
            latitude=40.7128,
            longitude=-74.0060
        )
        print(json.dumps(stores, indent=2))
    except Exception as e:
        print(f"Error finding stores: {str(e)}")
    
    # Test Google Pay integration
    print("\n3. Testing Google Pay API:")
    try:
        payment = api_manager.google_pay.create_payment_token(amount=99.99)
        print(json.dumps(payment, indent=2))
    except Exception as e:
        print(f"Error creating payment token: {str(e)}")

if __name__ == "__main__":
    test_api_integrations() 