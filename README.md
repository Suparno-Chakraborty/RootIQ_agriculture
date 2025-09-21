# Precision Agriculture Advisor

A comprehensive AI-powered farming advisory system that provides real-time crop health monitoring, weather intelligence, soil analysis, and personalized recommendations for modern precision agriculture.

## Features

### Advanced Crop Health Monitoring
- **Real-time satellite imagery analysis** with AI-powered crop health assessment
- **NDVI (Normalized Difference Vegetation Index)** calculation for vegetation health
- **Leaf Area Index** and **Chlorophyll content** monitoring
- **Growth stage estimation** and progress tracking
- **Field coverage analysis** with issue detection

### Comprehensive Weather Intelligence
- **Current weather conditions** with detailed metrics
- **7-day detailed forecasts** including temperature, humidity, and precipitation
- **Hourly weather tracking** for precise planning
- **Weather impact analysis** on crop health and growth
- **Interactive weather dashboards** with multiple visualization panels

### Smart Soil Health Analysis
- **Multi-parameter soil assessment** (pH, organic matter, soil type)
- **Water retention and drainage analysis**
- **Fertility evaluation** with improvement recommendations
- **Soil health scoring** and trend analysis
- **Customized soil management plans**

### AI-Powered Recommendations
- **Personalized irrigation scheduling** based on crop needs and weather
- **Fertilizer optimization** with growth stage-specific advice
- **Pest and disease management** with risk assessment
- **Market timing recommendations** for optimal profitability
- **Priority-based action plans** with immediate alerts

### Market Intelligence
- **Real-time price tracking** for major crops
- **Price trend analysis** with historical data
- **Revenue forecasting** and optimization strategies
- **Market volatility assessment**
- **Optimal selling time recommendations**

### Interactive Analytics Dashboard
- **Multi-tab interface** for organized data presentation
- **Real-time data visualizations** using Plotly charts
- **Health trend analysis** with 30-day historical simulation
- **Weather pattern analysis** with forecasting
- **Market price trend charts**

## Supported Crops

- **Wheat** - Complete growing season management
- **Rice** - Specialized irrigation and nutrient management
- **Corn** - Growth optimization and market timing
- **Soybeans** - Integrated pest management and yield maximization

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Data Visualization**: Plotly (Interactive charts and graphs)
- **Image Processing**: PIL (Python Imaging Library)
- **Data Analysis**: Pandas, NumPy
- **API Integration**: OpenWeatherMap API support
- **Machine Learning**: Simulated AI models for crop health analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/precision-agriculture-advisor.git
cd precision-agriculture-advisor
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Set up API keys** (Optional)
   - Get a free API key from [OpenWeatherMap](https://openweathermap.org/api)
   - Replace `YOUR_API_KEY_HERE` in the code with your actual API key
   - For demo purposes, the app works with simulated weather data

5. **Run the application**
```bash
streamlit run agriculture_advisor.py
```

6. **Access the application**
   - Open your web browser and go to `http://localhost:8501`

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
plotly>=5.15.0
```

## Usage

### 1. Farm Configuration
- Select your crop type from the dropdown menu
- Enter your farm's GPS coordinates (latitude and longitude)
- Configure soil parameters (type, pH, organic matter percentage)
- Set field information (size, irrigation type)

### 2. Analysis Dashboard
The application provides five main analysis tabs:

#### **Overview Tab**
- Quick health metrics and key performance indicators
- Satellite imagery analysis
- 30-day health trend visualization
- Immediate action alerts

#### **Detailed Analysis Tab**
- Comprehensive crop health metrics
- Soil analysis results with improvement recommendations
- Growth stage tracking and progress monitoring
- Issue detection and diagnostic information

#### **Weather Insights Tab**
- Current weather conditions with detailed metrics
- Interactive weather dashboard with multiple panels
- 7-day detailed forecast table
- Weather impact analysis on crop health

#### **Recommendations Tab**
- Categorized recommendations (Water, Nutrients, Pest Control, Timing)
- Priority-based action plans
- Growth stage-specific advice
- Customized farming strategies

#### **Market Analysis Tab**
- Current market prices and trends
- Revenue forecasting and optimization
- Price trend charts with historical data
- Market timing recommendations

### 3. Sample Locations
Try these pre-configured locations for testing:
- **Punjab Wheat Belt**: 30.9010, 75.8573
- **West Bengal Rice Fields**: 22.5726, 88.3639
- **Maharashtra Corn Belt**: 19.7515, 75.7139
- **Madhya Pradesh Soybean Zone**: 23.2599, 77.4126

## API Integration

### OpenWeatherMap API
- **Current Weather**: Real-time weather conditions
- **Weather Forecast**: 5-day/3-hour forecast data
- **Historical Weather**: Weather trend analysis

### Satellite Imagery API (Placeholder)
- Currently uses simulated satellite imagery
- Ready for integration with satellite data providers
- Supports real-time crop health monitoring

## Data Models

### Crop Health Analysis
- **Health Score**: Overall crop condition (0-100%)
- **NDVI Average**: Vegetation health index
- **Leaf Area Index**: Canopy density measurement
- **Chlorophyll Content**: Plant nutrition indicator
- **Growth Stage**: Current development phase

### Weather Analysis
- **Temperature Trends**: Historical and forecast data
- **Humidity Patterns**: Moisture level tracking
- **Precipitation Probability**: Rain forecast accuracy
- **Wind Conditions**: Speed and direction analysis

### Soil Assessment
- **pH Levels**: Soil acidity/alkalinity measurement
- **Organic Matter**: Soil fertility indicator
- **Water Retention**: Moisture holding capacity
- **Drainage Assessment**: Water flow characteristics

## Contributing

We welcome contributions to improve the Precision Agriculture Advisor! Here's how you can help:

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution
- **Real satellite API integration** (Sentinel, Landsat, Planet)
- **Machine learning model integration** for crop health prediction
- **Additional crop types** and region-specific data
- **Mobile app development** using React Native or Flutter
- **IoT sensor integration** for real-time field monitoring
- **Multi-language support** for international users

## Roadmap

### Version 2.0 (Planned)
- [ ] Real satellite imagery integration
- [ ] Machine learning-based crop disease detection
- [ ] IoT sensor data integration
- [ ] Multi-farm management dashboard
- [ ] Mobile application
- [ ] Advanced yield prediction models

### Version 2.1 (Future)
- [ ] Drone imagery integration
- [ ] Automated irrigation system control
- [ ] Supply chain management features
- [ ] Financial planning and budgeting tools
- [ ] Community features and knowledge sharing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Documentation
- **User Guide**: Comprehensive usage instructions
- **API Documentation**: Integration guidelines
- **Troubleshooting**: Common issues and solutions

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community forum for questions and ideas
- **Wiki**: Additional documentation and tutorials

### Contact
- **Email**: contact@hyprdevs.com
- **LinkedIn**: https://linkedin.com/company/hyprdevs

## Acknowledgments

- **OpenWeatherMap** for weather data API
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **Agricultural research community** for domain knowledge
- **Open source contributors** who make projects like this possible

## Screenshots

### Main Dashboard
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1b4b8876-410c-4c88-aa60-ac29a91da70f" />


### Weather Analysis
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a7027386-6332-4c37-a23a-1229be65dbcb" />


### Market Analysis
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/412fcc15-e30c-4dfb-8cd7-7d4a40dfa5be" />




---
