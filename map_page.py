import streamlit as st
import folium
from streamlit_folium import st_folium
import random

def generate_random_stores(center_lat, center_lng, num_stores=5, radius_km=10):
    """Generate random store locations within a radius of the center point"""
    stores = []
    for i in range(num_stores):
        # Convert radius from km to degrees (approximate)
        radius_lat = radius_km / 111  # 1 degree latitude = 111 km
        radius_lng = radius_km / (111 * abs(center_lat / 90))  # Adjust for latitude
        
        # Generate random offset
        lat = center_lat + random.uniform(-radius_lat, radius_lat)
        lng = center_lng + random.uniform(-radius_lng, radius_lng)
        
        store_name = f"Quick Basket Store #{i+1}"
        store_type = random.choice(["Flagship Store", "Express Store", "Outlet Store"])
        stores.append({
            "name": store_name,
            "type": store_type,
            "location": [lat, lng],
            "features": random.sample([
                "Nike Collection", 
                "Adidas Collection", 
                "Puma Collection",
                "Click & Collect",
                "Shoe Fitting",
                "Express Delivery"
            ], k=3)
        })
    return stores

def create_store_map():
    st.title("Quick Basket Store Locations")
    
    # Mumbai coordinates
    mumbai_center = [19.0760, 72.8777]
    
    # Generate random store locations
    stores = generate_random_stores(
        center_lat=mumbai_center[0],
        center_lng=mumbai_center[1],
        num_stores=8,  # Number of random stores
        radius_km=10   # Within 10km radius
    )
    
    # Create the base map
    m = folium.Map(location=mumbai_center, zoom_start=12)
    
    # Add HQ marker
    folium.Marker(
        mumbai_center,
        popup="Quick Basket HQ - Mumbai",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add store markers with custom icons and popups
    for store in stores:
        # Create detailed popup HTML
        popup_html = f"""
        <div style='width: 200px'>
            <h4>{store['name']}</h4>
            <p><strong>Type:</strong> {store['type']}</p>
            <p><strong>Features:</strong></p>
            <ul>
                {''.join(f'<li>{feature}</li>' for feature in store['features'])}
            </ul>
        </div>
        """
        
        # Add marker with custom icon
        folium.Marker(
            store['location'],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(
                color='blue',
                icon='shopping-cart',
                prefix='fa'
            )
        ).add_to(m)
    
    # Add circle showing delivery radius
    folium.Circle(
        mumbai_center,
        radius=10000,  # 10km in meters
        color='green',
        fill=True,
        popup='Delivery Zone'
    ).add_to(m)
    
    # Display the map
    st_folium(m, width=700, height=500)
    
    # Display store list
    st.subheader("Store Directory")
    for store in stores:
        with st.expander(f"{store['name']} - {store['type']}"):
            st.write("**Features:**")
            for feature in store['features']:
                st.write(f"- {feature}")
            st.write(f"**Location:** {store['location'][0]:.4f}, {store['location'][1]:.4f}")

if __name__ == "__main__":
    create_store_map() 