# NexGen Logistics - Delivery Delay Prediction System

A comprehensive data-driven Streamlit application for predicting delivery delays and optimizing logistics operations.

## 🎯 Problem Statement

NexGen Logistics is facing delivery delays caused by traffic, carrier issues, and inefficient fleet management. This application helps predict which orders are at risk of delay and suggests corrective actions.

## ✨ Features

### 📈 Overview Dashboard
- **Real-time KPIs**: Total orders, on-time delivery percentage, average delay, customer ratings
- **Financial Metrics**: Revenue, profit, cost per kilometer
- **Interactive Visualizations**:
  - Delivery status distribution (pie chart)
  - Carrier performance comparison
  - Distance vs delay analysis
  - Time series trends
  - Quality issues breakdown
  - Customer rating distribution

### 🎯 Delay Prediction
- **Machine Learning Model**: Random Forest Classifier trained on historical data
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC
- **Feature Importance**: Identify key factors affecting delays
- **Interactive Predictor**: Input order details to get real-time delay risk assessment
- **Smart Recommendations**: 
  - Route optimization suggestions
  - Vehicle assignment recommendations
  - Carrier alternatives
  - Traffic mitigation strategies
- **At-Risk Orders**: Identify current orders with high delay probability

### 📊 Deep Analytics
- **Trends Analysis**:
  - Weekly and monthly performance trends
  - Day-of-week patterns
  - Revenue vs cost analysis
  
- **Route Intelligence**:
  - Top routes by volume
  - Route performance metrics
  - Origin and destination insights
  - Traffic impact analysis
  
- **Financial Performance**:
  - Cost breakdown by category
  - Profitability by customer segment
  - Product category analysis
  - Cost efficiency metrics
  
- **Carrier & Quality Performance**:
  - Multi-dimensional carrier comparison
  - Priority level performance
  - Quality issue impact analysis
  - Feature correlation heatmap

### 🔍 Order Lookup
- Search by Order ID, Date Range, Customer Segment, or Route
- Comprehensive order details
- Cost breakdown visualization
- Performance metrics for individual orders

## 📊 Data Sources

The application uses the following CSV files:

| File | Key Information |
|------|----------------|
| `orders.csv` | Order ID, date, priority, value, category, origin, destination |
| `delivery_performance.csv` | Carrier, delivery times, status, quality issues, ratings |
| `routes_distance.csv` | Route details, distance, fuel consumption, traffic delays |
| `vehicle_fleet.csv` | Vehicle types, efficiency, location, status |
| `cost_breakdown.csv` | Fuel, labor, maintenance, and operational costs |


## 🎮 How to Use

### Dashboard Navigation

1. **Select Page** from the sidebar:
   - 📈 Overview Dashboard
   - 🎯 Delay Prediction
   - 📊 Deep Analytics
   - 🔍 Order Lookup

2. **Apply Filters**:
   - Date Range
   - Priority (Express, Standard, Economy)
   - Carrier (SpeedyLogistics, QuickShip, GlobalTransit)

### Making Predictions

1. Go to **🎯 Delay Prediction** page
2. Scroll to "Predict Delay Risk for New Orders"
3. Enter order details:
   - Distance, fuel consumption, traffic delay
   - Order value, priority, toll charges
   - Carrier, day of week, month
4. Click **"Predict Delay Risk"**
5. View:
   - Delay probability
   - Risk gauge
   - Actionable recommendations
   - Suggested vehicles

### Analyzing Performance

1. Go to **📊 Deep Analytics** page
2. Explore different tabs:
   - **Trends**: Time-based performance patterns
   - **Routes**: Geographic and route analysis
   - **Financial**: Cost and profitability insights
   - **Performance**: Carrier and quality metrics

### Looking Up Orders

1. Go to **🔍 Order Lookup** page
2. Choose search method
3. Enter search criteria
4. View detailed order information and metrics

## 🧠 Machine Learning Model

### Algorithm
- **Random Forest Classifier** with 100 estimators
- Balanced class weights to handle imbalanced data

### Features Used
- Distance (KM)
- Fuel consumption (L)
- Toll charges (INR)
- Traffic delay (minutes)
- Priority score
- Order value (INR)
- Total cost
- Cost per kilometer
- Fuel efficiency
- Day of week
- Month
- Carrier reliability
- Route complexity

### Model Performance
- Training/Test Split: 80/20
- Cross-validation with stratification
- Performance metrics: Accuracy, Precision, Recall, F1, ROC AUC

## 📈 Key Insights & Metrics

### KPIs Tracked
- ✅ On-Time Delivery Percentage
- ⏱️ Average Delay (days)
- ⭐ Customer Rating (1-5)
- 💰 Revenue and Profit
- 🚗 Cost per Kilometer
- ⚠️ Delayed Orders Count

### Delay Risk Factors
1. **Distance**: Longer routes have higher delay risk
2. **Traffic**: Heavy traffic significantly impacts delays
3. **Carrier Reliability**: Historical carrier performance matters
4. **Priority Level**: Express orders need special handling
5. **Route Complexity**: Combined measure of distance, traffic, and fuel consumption

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects
- **Machine Learning**: Scikit-learn
- **Modeling**: Random Forest, Logistic Regression (optional)

## 📝 Feature Engineering

The application performs extensive feature engineering:

1. **Delay Metrics**: delay_days, on_time, delay_category
2. **Cost Metrics**: total_cost, cost_per_km, profit_margin
3. **Efficiency Metrics**: fuel_efficiency, cost_efficiency
4. **Traffic Impact**: traffic_hours, high_traffic indicator
5. **Date Features**: day_of_week, month, week_of_year
6. **Risk Factors**: distance_risk, route_complexity
7. **Quality Metrics**: has_quality_issue, carrier_reliability

## 🎨 Customization

### Modify Filters
Edit the sidebar section in `app.py` to add/remove filters

### Add New Visualizations
Add charts in the respective page functions:
- `show_overview()` - Overview page
- `show_prediction()` - Prediction page
- `show_analytics()` - Analytics page
- `show_order_lookup()` - Lookup page

### Adjust Model Parameters
Modify the `train_model()` function in the prediction section

## 🐛 Troubleshooting

### Issue: Data Not Loading
- Verify CSV files are in the `dataset/` folder
- Check file names match exactly (case-sensitive)

### Issue: Module Not Found
```powershell
pip install -r requirements.txt --upgrade
```

### Issue: Slow Performance
- Reduce date range in filters
- Clear Streamlit cache: `streamlit cache clear`

### Issue: Port Already in Use
```powershell
streamlit run app.py --server.port 8502
```

## 🚀 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Geospatial route visualization
- [ ] Automated email alerts for at-risk orders
- [ ] Integration with fleet management systems
- [ ] Weather data incorporation
- [ ] Driver performance tracking
- [ ] Demand forecasting

## 📄 License

This project is created for educational and business analytics purposes.
