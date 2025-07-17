import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Load and prepare the data
# ------------------------------
st.set_page_config(page_title="🏠 Home Price Predictor", layout="centered")
st.title("🏠 Home Price Predictor")

# Load dataset
data = pd.read_csv("D:\RahulKarthik\kc_house_data.csv")

# Encode ZIP codes
le = LabelEncoder()
data['zipcode_encoded'] = le.fit_transform(data['zipcode'])

# Features and target
features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition',
            'grade', 'view', 'zipcode_encoded', 'yr_built', 'yr_renovated']
target = 'price'

# Prepare training data
X = data[features]
y = data[target]

# Train model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("📋 Enter House Details")

bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1.0, 8.0, 2.0, step=0.25)
sqft_living = st.slider("Sqft Living", 500, 6000, 1800)
floors = st.slider("Floors", 1.0, 4.0, 1.0, step=0.5)
condition = st.slider("Condition (1–5)", 1, 5, 3)
grade = st.slider("Grade (1–13)", 1, 13, 7)
view = st.slider("View (0–4)", 0, 4, 0)

# ZIP Code Selection
zip_display = sorted(data['zipcode'].unique())
selected_zip = st.selectbox("📮 Select ZIP Code (U.S.)", zip_display)
encoded_zip = le.transform([selected_zip])[0]

# Year Built & Renovated
st.markdown("#### 🏗️ Construction & Renovation")
yr_built = st.slider("Year Built", 1900, 2022, 2000)
yr_renovated = st.slider("Year Renovated", 0, 2022, 0, help="Use 0 if never renovated")

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔮 Predict Price"):
    input_data = np.array([[bedrooms, bathrooms, sqft_living, floors, condition,
                            grade, view, encoded_zip, yr_built, yr_renovated]])
    prediction = model.predict(input_data)[0]
    price_inr = prediction * 83  # USD to INR

    st.success(f"💰 Estimated Price: ₹{price_inr:,.0f}")
    st.info(f"🏗️ Built in: {yr_built}, 🔧 Renovated in: {yr_renovated if yr_renovated != 0 else 'Never'}")

# ------------------------------
# 📈 Price Trend Chart Section
# ------------------------------
st.markdown("---")
st.header("📊 House Price Trends Over Time")

# Dropdown to choose Year Built or Renovated
chart_option = st.selectbox("View trend by:", ["Year Built", "Year Renovated"])
agg_method = st.radio("Aggregation method:", ["Mean", "Median"])

# Determine which field to use
if chart_option == "Year Built":
    x_field = 'yr_built'
else:
    x_field = 'yr_renovated'

# Filter and group data
if x_field == 'yr_renovated':
    data_filtered = data[data['yr_renovated'] > 0]
else:
    data_filtered = data

# Aggregate
if agg_method == "Mean":
    chart_data = data_filtered.groupby(x_field)['price'].mean().reset_index().rename(columns={'price': 'value'})
else:
    chart_data = data_filtered.groupby(x_field)['price'].median().reset_index().rename(columns={'price': 'value'})

# Build chart with regression trendline
chart = alt.Chart(chart_data).mark_line(point=True, color="blue").encode(
    x=alt.X(x_field + ':O', title=chart_option),
    y=alt.Y('value:Q', title=f'{agg_method} Price (USD)'),
    tooltip=[x_field, 'value']
) + alt.Chart(chart_data).transform_regression(x_field, 'value').mark_line(color='red').encode(
    x=x_field + ':O',
    y='value:Q'
)

st.altair_chart(chart, use_container_width=True)

# ------------------------------
# 📥 Download CSV
# ------------------------------
st.download_button(
    label="⬇️ Download Chart Data as CSV",
    data=chart_data.to_csv(index=False).encode('utf-8'),
    file_name=f"{chart_option.replace(' ', '_').lower()}_price_trend.csv",
    mime='text/csv'
)
