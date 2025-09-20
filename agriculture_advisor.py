import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from PIL import Image
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Precision Agriculture Advisor",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2E8B57;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #daa520;
        margin: 0.5rem 0;
    }
    .alert-card {
        background-color: #ffe4e1;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 0.5rem 0;
    }
    .health-score-excellent {
        color: #228B22;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .health-score-good {
        color: #32CD32;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .health-score-fair {
        color: #FFA500;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .health-score-poor {
        color: #FF4500;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .pest-risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .pest-risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .pest-risk-high {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constants and Configuration
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual API key
SATELLITE_API_KEY = "YOUR_SATELLITE_API_KEY"

# Enhanced crop data with disease patterns and growth stages
CROP_DATA = {
    "wheat": {
        "growing_season": 120,
        "planting_months": [10, 11, 12],
        "harvest_months": [4, 5, 6],
        "water_requirement": "moderate",
        "fertilizer_schedule": ["planting", "30_days", "60_days"],
        "growth_stages": ["germination", "tillering", "jointing", "flowering", "grain_filling", "maturity"],
        "common_diseases": ["rust", "blight", "smut"],
        "optimal_temp_range": (15, 25),
        "optimal_humidity_range": (50, 70),
        "ph_range": (6.0, 7.5)
    },
    "rice": {
        "growing_season": 150,
        "planting_months": [6, 7, 8],
        "harvest_months": [11, 12, 1],
        "water_requirement": "high",
        "fertilizer_schedule": ["planting", "20_days", "40_days", "60_days"],
        "growth_stages": ["seedling", "tillering", "stem_elongation", "panicle_initiation", "heading", "maturity"],
        "common_diseases": ["blast", "bacterial_blight", "brown_spot"],
        "optimal_temp_range": (20, 30),
        "optimal_humidity_range": (70, 90),
        "ph_range": (5.5, 6.5)
    },
    "corn": {
        "growing_season": 100,
        "planting_months": [3, 4, 5],
        "harvest_months": [8, 9, 10],
        "water_requirement": "moderate",
        "fertilizer_schedule": ["planting", "30_days", "60_days"],
        "growth_stages": ["emergence", "vegetative", "tasseling", "silking", "grain_filling", "maturity"],
        "common_diseases": ["corn_borer", "rust", "smut"],
        "optimal_temp_range": (18, 28),
        "optimal_humidity_range": (55, 75),
        "ph_range": (6.0, 7.0)
    },
    "soybeans": {
        "growing_season": 110,
        "planting_months": [4, 5, 6],
        "harvest_months": [9, 10, 11],
        "water_requirement": "moderate",
        "fertilizer_schedule": ["planting", "45_days"],
        "growth_stages": ["emergence", "vegetative", "flowering", "pod_development", "seed_filling", "maturity"],
        "common_diseases": ["root_rot", "rust", "mosaic_virus"],
        "optimal_temp_range": (20, 30),
        "optimal_humidity_range": (60, 80),
        "ph_range": (6.0, 7.0)
    }
}

# Enhanced market price data with historical trends
MARKET_PRICES = {
    "wheat": {
        "current": 250, "trend": "up", "forecast_30d": 265,
        "historical": [240, 245, 248, 250, 252],
        "volatility": "low"
    },
    "rice": {
        "current": 180, "trend": "stable", "forecast_30d": 185,
        "historical": [175, 178, 180, 182, 180],
        "volatility": "low"
    },
    "corn": {
        "current": 220, "trend": "down", "forecast_30d": 210,
        "historical": [235, 230, 225, 220, 218],
        "volatility": "medium"
    },
    "soybeans": {
        "current": 400, "trend": "up", "forecast_30d": 420,
        "historical": [380, 390, 395, 400, 405],
        "volatility": "high"
    }
}

# Soil type data
SOIL_TYPES = {
    "clay": {"water_retention": "high", "drainage": "poor", "fertility": "high"},
    "loam": {"water_retention": "medium", "drainage": "good", "fertility": "high"},
    "sandy": {"water_retention": "low", "drainage": "excellent", "fertility": "medium"},
    "silt": {"water_retention": "high", "drainage": "medium", "fertility": "medium"}
}

def generate_enhanced_satellite_image():
    """Generate a more detailed satellite image for demo purposes"""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Base field (healthy green)
    img[50:450, 50:450] = [34, 139, 34]
    
    # Add field sections with varying health
    sections = [
        ([60, 240, 60, 240], [50, 205, 50]),    # Healthy section
        ([60, 240, 260, 440], [85, 107, 47]),   # Stressed section
        ([260, 440, 60, 240], [124, 252, 0]),   # Very healthy
        ([260, 440, 260, 440], [107, 142, 35])  # Average health
    ]
    
    for (y1, y2, x1, x2), color in sections:
        img[y1:y2, x1:x2] = color
    
    # Add water source
    img[400:440, 400:440] = [30, 144, 255]
    
    # Add roads/paths
    img[245:255, 50:450] = [139, 69, 19]
    img[50:450, 245:255] = [139, 69, 19]
    
    # Add some problematic areas (disease/pest damage)
    for _ in range(5):
        x, y = random.randint(70, 430), random.randint(70, 430)
        size = random.randint(10, 25)
        img[y:y+size, x:x+size] = [160, 82, 45]  # Brown spots
    
    return Image.fromarray(img)

def analyze_soil_health(soil_type, ph_level, organic_matter):
    """Analyze soil health based on parameters"""
    soil_info = SOIL_TYPES.get(soil_type, SOIL_TYPES["loam"])
    
    # Calculate soil health score
    ph_score = 1.0 if 6.0 <= ph_level <= 7.5 else max(0.3, 1.0 - abs(ph_level - 6.75) / 2.0)
    organic_score = min(1.0, organic_matter / 3.0)  # 3% is considered good
    
    overall_score = (ph_score + organic_score) / 2
    
    recommendations = []
    if ph_level < 6.0:
        recommendations.append("Apply lime to increase soil pH")
    elif ph_level > 7.5:
        recommendations.append("Apply sulfur to decrease soil pH")
    
    if organic_matter < 2.0:
        recommendations.append("Add compost or organic matter to improve soil structure")
    
    return {
        "health_score": overall_score,
        "ph_score": ph_score,
        "organic_score": organic_score,
        "recommendations": recommendations,
        "water_retention": soil_info["water_retention"],
        "drainage": soil_info["drainage"],
        "fertility": soil_info["fertility"]
    }

def simulate_enhanced_crop_health_analysis(crop_type, weather_data, soil_analysis):
    """Enhanced AI-based crop health analysis"""
    base_health = random.uniform(0.6, 0.95)
    
    # Weather impact
    current_temp = weather_data["current"]["main"]["temp"]
    current_humidity = weather_data["current"]["main"]["humidity"]
    
    crop_info = CROP_DATA[crop_type]
    temp_optimal = crop_info["optimal_temp_range"]
    humidity_optimal = crop_info["optimal_humidity_range"]
    
    # Temperature stress factor
    if temp_optimal[0] <= current_temp <= temp_optimal[1]:
        temp_factor = 0.05
    else:
        temp_factor = -0.1 * min(abs(current_temp - temp_optimal[0]), abs(current_temp - temp_optimal[1])) / 10
    
    # Humidity stress factor
    if humidity_optimal[0] <= current_humidity <= humidity_optimal[1]:
        humidity_factor = 0.02
    else:
        humidity_factor = -0.05 * min(abs(current_humidity - humidity_optimal[0]), abs(current_humidity - humidity_optimal[1])) / 20
    
    # Soil impact
    soil_factor = (soil_analysis["health_score"] - 0.5) * 0.1
    
    # Calculate final health score
    health_score = max(0.3, min(0.98, base_health + temp_factor + humidity_factor + soil_factor))
    
    # Generate detailed analysis
    issues = []
    if current_temp > temp_optimal[1] + 5:
        issues.append("Heat stress detected - temperature above optimal range")
    elif current_temp < temp_optimal[0] - 3:
        issues.append("Cold stress detected - temperature below optimal range")
    
    if current_humidity < humidity_optimal[0] - 10:
        issues.append("Moisture stress - humidity below optimal range")
    elif current_humidity > humidity_optimal[1] + 15:
        issues.append("High humidity may promote fungal diseases")
    
    if soil_analysis["health_score"] < 0.6:
        issues.append("Soil health issues detected")
    
    if health_score < 0.7:
        issues.extend(["Possible nutrient deficiency", "Consider pest inspection"])
    
    # Generate pest risk assessment
    pest_risk = calculate_pest_risk(crop_type, weather_data, health_score)
    
    # Growth stage estimation
    current_month = datetime.now().month
    growth_stage = estimate_growth_stage(crop_type, current_month)
    
    return {
        "health_score": health_score,
        "status": get_health_status(health_score),
        "issues": issues,
        "ndvi_avg": health_score * 0.8 + random.uniform(-0.05, 0.05),
        "coverage": random.uniform(0.85, 0.98),
        "pest_risk": pest_risk,
        "growth_stage": growth_stage,
        "leaf_area_index": health_score * 4.5 + random.uniform(-0.3, 0.3),
        "chlorophyll_content": health_score * 45 + random.uniform(-5, 5)
    }

def calculate_pest_risk(crop_type, weather_data, health_score):
    """Calculate pest risk based on weather and crop health"""
    temp = weather_data["current"]["main"]["temp"]
    humidity = weather_data["current"]["main"]["humidity"]
    
    # Base risk factors
    temp_risk = 0.3 if 20 <= temp <= 30 else 0.1
    humidity_risk = 0.4 if humidity > 70 else 0.2
    health_risk = 0.5 if health_score < 0.7 else 0.1
    
    total_risk = temp_risk + humidity_risk + health_risk
    
    if total_risk > 0.8:
        return {"level": "high", "score": total_risk}
    elif total_risk > 0.5:
        return {"level": "medium", "score": total_risk}
    else:
        return {"level": "low", "score": total_risk}

def estimate_growth_stage(crop_type, current_month):
    """Estimate current growth stage based on planting season"""
    crop_info = CROP_DATA[crop_type]
    planting_months = crop_info["planting_months"]
    stages = crop_info["growth_stages"]
    
    # Find the most recent planting month
    recent_planting = max([m for m in planting_months if m <= current_month] + 
                         [m - 12 for m in planting_months])
    
    months_since_planting = current_month - (recent_planting if recent_planting > 0 else recent_planting + 12)
    
    # Estimate stage based on time elapsed
    stage_duration = crop_info["growing_season"] / len(stages) / 30  # days to months
    stage_index = min(len(stages) - 1, int(months_since_planting / stage_duration))
    
    return stages[stage_index]

def get_health_status(score):
    """Convert health score to status"""
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.75:
        return "Good"
    elif score >= 0.65:
        return "Fair"
    else:
        return "Poor"

def fetch_weather_data(lat, lon, api_key):
    """Fetch weather data from OpenWeatherMap API"""
    # For demo purposes, always return simulated data
    return generate_enhanced_weather_data()

def generate_enhanced_weather_data():
    """Generate comprehensive weather data for demo"""
    current_temp = random.uniform(20, 35)
    current_humidity = random.uniform(40, 80)
    
    # Generate hourly data for today
    hourly_data = []
    for hour in range(24):
        hourly_data.append({
            "time": hour,
            "temp": current_temp + random.uniform(-3, 3),
            "humidity": current_humidity + random.uniform(-10, 10),
            "wind_speed": random.uniform(1, 8),
            "uv_index": max(0, 10 * np.sin(np.pi * hour / 12)) if 6 <= hour <= 18 else 0
        })
    
    return {
        "current": {
            "main": {
                "temp": current_temp,
                "humidity": current_humidity,
                "pressure": random.uniform(1000, 1020),
                "feels_like": current_temp + random.uniform(-2, 2)
            },
            "weather": [{"main": random.choice(["Clear", "Clouds", "Rain"]), 
                        "description": random.choice(["clear sky", "few clouds", "light rain"])}],
            "wind": {"speed": random.uniform(2, 8), "deg": random.randint(0, 360)},
            "uv": {"index": random.uniform(3, 9)},
            "name": "Farm Location"
        },
        "hourly": hourly_data,
        "forecast": {
            "list": [
                {
                    "dt": int((datetime.now() + timedelta(days=i)).timestamp()),
                    "main": {
                        "temp": current_temp + random.uniform(-5, 5),
                        "temp_min": current_temp + random.uniform(-8, -2),
                        "temp_max": current_temp + random.uniform(2, 8),
                        "humidity": current_humidity + random.uniform(-15, 15)
                    },
                    "weather": [{"main": random.choice(["Clear", "Clouds", "Rain"]), 
                               "description": random.choice(["clear sky", "few clouds", "moderate rain"])}],
                    "pop": random.uniform(0, 0.8),
                    "wind": {"speed": random.uniform(2, 10)}
                }
                for i in range(7)
            ]
        }
    }

def generate_enhanced_recommendations(crop_type, weather_data, health_analysis, soil_analysis):
    """Generate comprehensive farming recommendations"""
    recommendations = {
        "immediate": [],
        "irrigation": [],
        "fertilizer": [],
        "pest_control": [],
        "timing": [],
        "general": []
    }
    
    current_temp = weather_data["current"]["main"]["temp"]
    current_humidity = weather_data["current"]["main"]["humidity"]
    health_score = health_analysis["health_score"]
    pest_risk = health_analysis["pest_risk"]
    growth_stage = health_analysis["growth_stage"]
    
    # Immediate actions
    if health_score < 0.6:
        recommendations["immediate"].append("URGENT: Crop showing severe stress - immediate field inspection required")
    
    if pest_risk["level"] == "high":
        recommendations["immediate"].append(f"HIGH PEST RISK: Implement pest control measures immediately")
    
    # Irrigation recommendations
    crop_info = CROP_DATA[crop_type]
    if current_humidity < crop_info["optimal_humidity_range"][0]:
        recommendations["irrigation"].append("Increase irrigation frequency - humidity below optimal")
    elif current_humidity > crop_info["optimal_humidity_range"][1]:
        recommendations["irrigation"].append("Reduce irrigation - risk of waterlogging and fungal diseases")
    
    if soil_analysis["drainage"] == "poor" and weather_data["forecast"]["list"][0]["pop"] > 0.7:
        recommendations["irrigation"].append("Improve field drainage before expected rainfall")
    
    # Growth stage specific recommendations
    if growth_stage == "flowering":
        recommendations["irrigation"].append("Critical growth stage - maintain consistent soil moisture")
        recommendations["fertilizer"].append("Apply potassium-rich fertilizer to support flower development")
    elif growth_stage == "grain_filling":
        recommendations["irrigation"].append("Ensure adequate water supply during grain filling")
    
    # Fertilizer recommendations based on growth stage and soil health
    if soil_analysis["health_score"] < 0.7:
        recommendations["fertilizer"].extend(soil_analysis["recommendations"])
    
    if growth_stage in ["vegetative", "tillering"]:
        recommendations["fertilizer"].append("Apply nitrogen-rich fertilizer for vegetative growth")
    elif growth_stage == "flowering":
        recommendations["fertilizer"].append("Apply phosphorus for flower and fruit development")
    
    # Pest control recommendations
    if pest_risk["level"] == "high":
        recommendations["pest_control"].append("Deploy pheromone traps and increase field monitoring")
        recommendations["pest_control"].append("Consider integrated pest management approach")
    elif pest_risk["level"] == "medium":
        recommendations["pest_control"].append("Monitor field regularly for pest activity")
    
    if current_humidity > 80:
        recommendations["pest_control"].append("High humidity increases fungal disease risk - apply preventive fungicide")
    
    # Market-based timing recommendations
    market_info = MARKET_PRICES.get(crop_type, {})
    if market_info.get("trend") == "up":
        recommendations["timing"].append(f"Market prices trending upward - consider delayed harvest if crop maturity allows")
    elif market_info.get("trend") == "down":
        recommendations["timing"].append(f"Market prices declining - plan for timely harvest")
    
    # Weather-based recommendations
    upcoming_rain = any(day["pop"] > 0.6 for day in weather_data["forecast"]["list"][:3])
    if upcoming_rain:
        recommendations["general"].append("Rain expected in next 3 days - postpone field activities")
        recommendations["pest_control"].append("Apply treatments before rain if needed")
    
    return recommendations

def create_weather_charts(weather_data):
    """Create interactive weather visualization charts"""
    # Temperature and humidity forecast
    forecast_dates = []
    temps_max = []
    temps_min = []
    humidity_vals = []
    rain_prob = []
    
    for day in weather_data["forecast"]["list"]:
        date = datetime.fromtimestamp(day["dt"])
        forecast_dates.append(date.strftime("%m/%d"))
        temps_max.append(day["main"]["temp_max"])
        temps_min.append(day["main"]["temp_min"])
        humidity_vals.append(day["main"]["humidity"])
        rain_prob.append(day["pop"] * 100)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Forecast', 'Humidity Forecast', 
                       'Rain Probability', 'Hourly Temperature (Today)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature chart
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=temps_max, name="Max Temp", 
                  line=dict(color='red'), mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=temps_min, name="Min Temp", 
                  line=dict(color='blue'), mode='lines+markers'),
        row=1, col=1
    )
    
    # Humidity chart
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=humidity_vals, name="Humidity", 
                  line=dict(color='green'), mode='lines+markers'),
        row=1, col=2
    )
    
    # Rain probability
    fig.add_trace(
        go.Bar(x=forecast_dates, y=rain_prob, name="Rain Probability",
               marker_color='lightblue'),
        row=2, col=1
    )
    
    # Hourly temperature (today)
    hourly_hours = [f"{h}:00" for h in range(0, 24, 3)]  # Every 3 hours
    hourly_temps = [weather_data["hourly"][h]["temp"] for h in range(0, 24, 3)]
    
    fig.add_trace(
        go.Scatter(x=hourly_hours, y=hourly_temps, name="Hourly Temp",
                  line=dict(color='orange'), mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Weather Analysis Dashboard")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Humidity (%)", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Rain Probability (%)", row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
    
    return fig

def create_health_trend_chart(health_score):
    """Create a health trend simulation chart"""
    # Simulate historical health data
    days = list(range(-30, 1))
    health_trend = []
    
    base_health = health_score
    for day in days:
        # Add some realistic variation
        variation = np.sin(day / 5) * 0.05 + random.uniform(-0.03, 0.03)
        daily_health = max(0.3, min(0.98, base_health + variation))
        health_trend.append(daily_health)
    
    dates = [(datetime.now() + timedelta(days=day)).strftime("%m/%d") for day in days]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=health_trend,
        mode='lines+markers',
        name='Crop Health Score',
        line=dict(color='green', width=2)
    ))
    
    # Add health zones
    fig.add_hline(y=0.85, line_dash="dash", line_color="darkgreen", 
                  annotation_text="Excellent (>85%)")
    fig.add_hline(y=0.75, line_dash="dash", line_color="green", 
                  annotation_text="Good (75-85%)")
    fig.add_hline(y=0.65, line_dash="dash", line_color="orange", 
                  annotation_text="Fair (65-75%)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Poor (<65%)")
    
    fig.update_layout(
        title="30-Day Crop Health Trend",
        xaxis_title="Date",
        yaxis_title="Health Score",
        height=400,
        yaxis=dict(range=[0.4, 1.0])
    )
    
    return fig

def create_market_price_chart(crop_type):
    """Create market price trend chart"""
    if crop_type not in MARKET_PRICES:
        return None
    
    market_data = MARKET_PRICES[crop_type]
    historical_prices = market_data["historical"]
    current_price = market_data["current"]
    forecast_price = market_data["forecast_30d"]
    
    # Create extended price series
    dates = [(datetime.now() - timedelta(days=30-i*7)).strftime("%m/%d") for i in range(5)]
    dates.append(datetime.now().strftime("%m/%d"))
    dates.append((datetime.now() + timedelta(days=30)).strftime("%m/%d"))
    
    prices = historical_prices + [current_price, forecast_price]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates[:-1],
        y=prices[:-1],
        mode='lines+markers',
        name='Historical Prices',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates[-2:],
        y=prices[-2:],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{crop_type.title()} Market Price Trend",
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Precision Agriculture Advisor</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered farming insights with comprehensive analysis and recommendations**")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Farm Configuration")
        
        # Crop selection
        crop_type = st.selectbox(
            "Select Crop Type:",
            options=list(CROP_DATA.keys()),
            format_func=lambda x: x.title()
        )
        
        # Location input
        st.subheader("Farm Location")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=28.6139, format="%.4f")
        with col2:
            longitude = st.number_input("Longitude", value=77.2090, format="%.4f")
        
        # Soil parameters
        st.subheader("Soil Analysis")
        soil_type = st.selectbox("Soil Type", options=list(SOIL_TYPES.keys()))
        ph_level = st.slider("Soil pH", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
        organic_matter = st.slider("Organic Matter (%)", min_value=0.5, max_value=8.0, value=2.5, step=0.1)
        
        # Field size and additional parameters
        st.subheader("Field Information")
        field_size = st.number_input("Field Size (acres)", min_value=0.1, value=5.0, step=0.1)
        irrigation_type = st.selectbox("Irrigation Type", 
                                     ["Drip", "Sprinkler", "Flood", "Rain-fed"])
        
        # Analysis button
        analyze_button = st.button("Analyze Farm", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button:
        st.markdown("---")
        
        with st.spinner("Analyzing farm data and generating insights..."):
            # Analyze soil health
            soil_analysis = analyze_soil_health(soil_type, ph_level, organic_matter)
            
            # Generate/fetch satellite image
            satellite_image = generate_enhanced_satellite_image()
            
            # Fetch weather data
            weather_data = fetch_weather_data(latitude, longitude, OPENWEATHER_API_KEY)
            
            # Analyze crop health
            health_analysis = simulate_enhanced_crop_health_analysis(crop_type, weather_data, soil_analysis)
            
            # Generate recommendations
            recommendations = generate_enhanced_recommendations(crop_type, weather_data, health_analysis, soil_analysis)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Detailed Analysis", "Weather Insights", "Recommendations", "Market Analysis"])
        
        with tab1:
            # Overview Dashboard
            st.subheader("Farm Overview Dashboard")
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                health_score = health_analysis["health_score"]
                health_status = health_analysis["status"]
                status_class = f"health-score-{health_status.lower()}"
                st.markdown(f'<div class="{status_class}">Overall Health</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{status_class}">{health_score:.1%}</div>', unsafe_allow_html=True)
            
            with col2:
                pest_risk_level = health_analysis["pest_risk"]["level"]
                risk_class = f"pest-risk-{pest_risk_level}"
                st.markdown(f'<div class="{risk_class}">Pest Risk</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{risk_class}">{pest_risk_level.title()}</div>', unsafe_allow_html=True)
            
            with col3:
                current_temp = weather_data["current"]["main"]["temp"]
                st.metric("Temperature", f"{current_temp:.1f}°C")
            
            with col4:
                current_humidity = weather_data["current"]["main"]["humidity"]
                st.metric("Humidity", f"{current_humidity:.0f}%")
            
            # Two column layout for overview
            over_col1, over_col2 = st.columns([1, 1])
            
            with over_col1:
                # Satellite Image
                st.subheader("Satellite Imagery Analysis")
                st.image(satellite_image, caption=f"Farm Location: {latitude:.4f}, {longitude:.4f}")
                
                # Quick stats
                st.write("**Field Analysis:**")
                st.write(f"- Field Size: {field_size} acres")
                st.write(f"- Growth Stage: {health_analysis['growth_stage'].replace('_', ' ').title()}")
                st.write(f"- Coverage: {health_analysis['coverage']:.1%}")
                st.write(f"- NDVI Average: {health_analysis['ndvi_avg']:.3f}")
                
            with over_col2:
                # Health trend chart
                health_chart = create_health_trend_chart(health_analysis["health_score"])
                st.plotly_chart(health_chart, use_container_width=True)
                
                # Immediate actions if any
                if recommendations["immediate"]:
                    st.subheader("Immediate Actions Required")
                    for action in recommendations["immediate"]:
                        st.markdown(f'<div class="alert-card">{action}</div>', unsafe_allow_html=True)
        
        with tab2:
            # Detailed Analysis
            st.subheader("Comprehensive Farm Analysis")
            
            # Crop Health Details
            detail_col1, detail_col2 = st.columns([1, 1])
            
            with detail_col1:
                st.subheader("Crop Health Metrics")
                
                # Health metrics in a nice layout
                metrics_data = {
                    "Metric": ["Health Score", "NDVI Average", "Leaf Area Index", "Chlorophyll Content", "Field Coverage"],
                    "Value": [
                        f"{health_analysis['health_score']:.1%}",
                        f"{health_analysis['ndvi_avg']:.3f}",
                        f"{health_analysis['leaf_area_index']:.2f}",
                        f"{health_analysis['chlorophyll_content']:.1f} mg/g",
                        f"{health_analysis['coverage']:.1%}"
                    ],
                    "Status": [
                        health_analysis["status"],
                        "Good" if health_analysis['ndvi_avg'] > 0.6 else "Fair",
                        "Optimal" if health_analysis['leaf_area_index'] > 3.0 else "Below Optimal",
                        "Good" if health_analysis['chlorophyll_content'] > 35 else "Low",
                        "Excellent" if health_analysis['coverage'] > 0.9 else "Good"
                    ]
                }
                
                st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
                
                # Issues detected
                if health_analysis["issues"]:
                    st.subheader("Issues Detected")
                    for issue in health_analysis["issues"]:
                        st.markdown(f'<div class="alert-card">{issue}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-card">No significant issues detected</div>', unsafe_allow_html=True)
            
            with detail_col2:
                st.subheader("Soil Analysis Results")
                
                # Soil health metrics
                soil_metrics = {
                    "Parameter": ["Overall Health", "pH Level", "Organic Matter", "Water Retention", "Drainage", "Fertility"],
                    "Value": [
                        f"{soil_analysis['health_score']:.1%}",
                        f"{ph_level:.1f}",
                        f"{organic_matter:.1f}%",
                        soil_analysis["water_retention"],
                        soil_analysis["drainage"],
                        soil_analysis["fertility"]
                    ],
                    "Rating": [
                        "Good" if soil_analysis['health_score'] > 0.7 else "Fair",
                        "Optimal" if 6.0 <= ph_level <= 7.5 else "Needs Adjustment",
                        "Good" if organic_matter >= 2.0 else "Low",
                        soil_analysis["water_retention"].title(),
                        soil_analysis["drainage"].title(),
                        soil_analysis["fertility"].title()
                    ]
                }
                
                st.dataframe(pd.DataFrame(soil_metrics), hide_index=True, use_container_width=True)
                
                # Soil recommendations
                if soil_analysis["recommendations"]:
                    st.subheader("Soil Improvement Recommendations")
                    for rec in soil_analysis["recommendations"]:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
                
                # Growth stage information
                st.subheader("Growth Stage Information")
                current_stage = health_analysis['growth_stage']
                crop_info = CROP_DATA[crop_type]
                stages = crop_info['growth_stages']
                
                stage_index = stages.index(current_stage) if current_stage in stages else 0
                progress = (stage_index + 1) / len(stages)
                
                st.progress(progress)
                st.write(f"**Current Stage:** {current_stage.replace('_', ' ').title()}")
                st.write(f"**Progress:** {progress:.1%} through growing season")
                st.write(f"**Next Stage:** {stages[min(stage_index + 1, len(stages) - 1)].replace('_', ' ').title()}")
        
        with tab3:
            # Weather Insights
            st.subheader("Weather Analysis and Forecasts")
            
            # Current weather summary
            current = weather_data["current"]
            st.subheader("Current Conditions")
            
            weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
            
            with weather_col1:
                st.metric("Temperature", f"{current['main']['temp']:.1f}°C",
                         f"Feels like {current['main']['feels_like']:.1f}°C")
            with weather_col2:
                st.metric("Humidity", f"{current['main']['humidity']:.0f}%")
            with weather_col3:
                st.metric("Wind Speed", f"{current['wind']['speed']:.1f} m/s")
            with weather_col4:
                st.metric("UV Index", f"{current.get('uv', {}).get('index', 'N/A')}")
            
            # Weather charts
            weather_chart = create_weather_charts(weather_data)
            st.plotly_chart(weather_chart, use_container_width=True)
            
            # 7-day forecast table
            st.subheader("7-Day Detailed Forecast")
            forecast_data = []
            for day in weather_data["forecast"]["list"]:
                date = datetime.fromtimestamp(day["dt"])
                forecast_data.append({
                    "Date": date.strftime("%A, %m/%d"),
                    "High/Low": f"{day['main']['temp_max']:.0f}°C / {day['main']['temp_min']:.0f}°C",
                    "Condition": day["weather"][0]["main"],
                    "Humidity": f"{day['main']['humidity']:.0f}%",
                    "Rain Chance": f"{day['pop']*100:.0f}%",
                    "Wind": f"{day['wind']['speed']:.1f} m/s"
                })
            
            st.dataframe(pd.DataFrame(forecast_data), hide_index=True, use_container_width=True)
        
        with tab4:
            # Recommendations
            st.subheader("Personalized Farming Recommendations")
            
            # Create columns for different recommendation types
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                # Irrigation recommendations
                if recommendations["irrigation"]:
                    st.markdown("#### Water Management")
                    for rec in recommendations["irrigation"]:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
                
                # Fertilizer recommendations
                if recommendations["fertilizer"]:
                    st.markdown("#### Nutrient Management")
                    for rec in recommendations["fertilizer"]:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
                
                # Timing recommendations
                if recommendations["timing"]:
                    st.markdown("#### Timing & Scheduling")
                    for rec in recommendations["timing"]:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
            
            with rec_col2:
                # Pest control recommendations
                if recommendations["pest_control"]:
                    st.markdown("#### Pest & Disease Management")
                    for rec in recommendations["pest_control"]:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
                
                # General recommendations
                if recommendations["general"]:
                    st.markdown("#### General Recommendations")
                    for rec in recommendations["general"]:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
            
            # Action plan
            st.markdown("---")
            st.subheader("Recommended Action Plan")
            
            action_plan = []
            if recommendations["immediate"]:
                action_plan.extend([("Immediate", action, "High") for action in recommendations["immediate"]])
            
            # Add other recommendations with priorities
            for category, recs in recommendations.items():
                if category != "immediate" and recs:
                    priority = "High" if category == "pest_control" else "Medium"
                    for rec in recs[:2]:  # Limit to top 2 per category
                        action_plan.append((category.replace("_", " ").title(), rec, priority))
            
            if action_plan:
                action_df = pd.DataFrame(action_plan, columns=["Category", "Action", "Priority"])
                st.dataframe(action_df, hide_index=True, use_container_width=True)
        
        with tab5:
            # Market Analysis
            st.subheader("Market Price Analysis")
            
            if crop_type in MARKET_PRICES:
                market_info = MARKET_PRICES[crop_type]
                
                # Current market status
                market_col1, market_col2, market_col3, market_col4 = st.columns(4)
                
                with market_col1:
                    st.metric("Current Price", f"₹{market_info['current']}/quintal")
                
                with market_col2:
                    trend_indicator = "↑" if market_info['trend'] == 'up' else "↓" if market_info['trend'] == 'down' else "→"
                    st.metric("Trend", f"{trend_indicator} {market_info['trend'].title()}")
                
                with market_col3:
                    change = market_info['forecast_30d'] - market_info['current']
                    st.metric("30-Day Forecast", f"₹{market_info['forecast_30d']}/quintal", f"{change:+.0f}")
                
                with market_col4:
                    st.metric("Volatility", market_info['volatility'].title())
                
                # Price trend chart
                price_chart = create_market_price_chart(crop_type)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                
                # Market insights
                st.subheader("Market Insights")
                
                potential_revenue = field_size * 25 * market_info['current']  # Assuming 25 quintals/acre
                forecast_revenue = field_size * 25 * market_info['forecast_30d']
                revenue_change = forecast_revenue - potential_revenue
                
                col_rev1, col_rev2, col_rev3 = st.columns(3)
                
                with col_rev1:
                    st.metric("Current Potential Revenue", f"₹{potential_revenue:,.0f}")
                with col_rev2:
                    st.metric("Forecast Revenue (30d)", f"₹{forecast_revenue:,.0f}")
                with col_rev3:
                    st.metric("Expected Change", f"₹{revenue_change:+,.0f}")
                
                # Market recommendations
                if market_info['trend'] == 'up':
                    st.markdown('<div class="success-card">Market Opportunity: Prices are trending upward. Consider delaying harvest if crop maturity allows.</div>', unsafe_allow_html=True)
                elif market_info['trend'] == 'down':
                    st.markdown('<div class="alert-card">Price Alert: Market prices are declining. Plan for timely harvest to avoid further losses.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="recommendation-card">Market Status: Prices are stable. Proceed with regular harvest schedule.</div>', unsafe_allow_html=True)
    
    else:
        # Welcome message and feature showcase
        st.markdown("---")
        st.info("Configure your farm details in the sidebar and click 'Analyze Farm' to get comprehensive insights!")
        
        # Enhanced feature showcase
        st.subheader("Advanced Agriculture Analytics Platform")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Satellite & AI Analysis
            - Real-time crop health monitoring
            - NDVI vegetation index analysis
            - Growth stage estimation
            - Disease and pest detection
            - Field coverage assessment
            """)
        
        with col2:
            st.markdown("""
            ### Weather Intelligence
            - Current conditions monitoring
            - 7-day detailed forecasts
            - Hourly weather tracking
            - Risk assessment alerts
            - Climate impact analysis
            """)
        
        with col3:
            st.markdown("""
            ### Smart Recommendations
            - Personalized irrigation scheduling
            - Fertilizer optimization
            - Pest control strategies
            - Market timing advice
            - Soil health improvement
            """)
        
        # Additional features section
        st.markdown("---")
        st.subheader("Comprehensive Farm Management Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            ### Soil Analysis & Management
            - Detailed soil health assessment
            - pH and nutrient level monitoring
            - Organic matter analysis
            - Drainage and fertility evaluation
            - Customized improvement plans
            
            ### Growth Monitoring
            - Growth stage tracking
            - Development progress analysis
            - Yield prediction models
            - Quality assessment indicators
            """)
        
        with feature_col2:
            st.markdown("""
            ### Market Intelligence
            - Real-time price tracking
            - Trend analysis and forecasting
            - Revenue optimization suggestions
            - Volatility risk assessment
            - Optimal selling strategies
            
            ### Risk Management
            - Pest risk evaluation
            - Disease prediction models
            - Weather impact assessment
            - Early warning systems
            """)
        
        # Sample locations with enhanced details
        st.markdown("---")
        st.subheader("Try These Sample Farm Locations")
        
        sample_locations = [
            {
                "name": "Punjab Wheat Belt", 
                "lat": 30.9010, "lon": 75.8573, 
                "crop": "wheat",
                "description": "Prime wheat growing region with optimal soil conditions"
            },
            {
                "name": "West Bengal Rice Fields", 
                "lat": 22.5726, "lon": 88.3639, 
                "crop": "rice",
                "description": "Traditional rice cultivation area with monsoon irrigation"
            },
            {
                "name": "Maharashtra Corn Belt", 
                "lat": 19.7515, "lon": 75.7139, 
                "crop": "corn",
                "description": "Modern corn farming with advanced irrigation systems"
            },
            {
                "name": "Madhya Pradesh Soybean Zone", 
                "lat": 23.2599, "lon": 77.4126, 
                "crop": "soybeans",
                "description": "Major soybean producing region with favorable climate"
            }
        ]
        
        for location in sample_locations:
            with st.expander(f"{location['name']} - {location['crop'].title()}"):
                st.write(f"**Coordinates:** {location['lat']}, {location['lon']}")
                st.write(f"**Description:** {location['description']}")
                st.write(f"**Recommended Analysis:** Use these coordinates for {location['crop']} crop analysis")

if __name__ == "__main__":
    main()