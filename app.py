import streamlit as st
import pandas as pd
import statsmodels.api as sm

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('yield_df.csv', index_col=0)
    df = df.drop_duplicates()
    return df

df = load_data()
df_original = df.copy()

# Prepare data for modeling
# Select only numeric columns and target before one-hot encoding
numeric_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
df_numeric = df[numeric_cols].copy()

# One-hot encode categorical variables
dummies_area = pd.get_dummies(df['Area'], prefix='Area')
dummies_item = pd.get_dummies(df['Item'], prefix='Item')

# Concatenate numeric and dummy variables
df = pd.concat([df_numeric, dummies_area, dummies_item], axis=1)

# Handle missing values and convert to numeric
if df.isnull().any().any():
    st.warning("Found missing values in the dataset. Filling with 0 for now.")
    df = df.fillna(0)

# Ensure all columns are numeric
df = df.astype(float)

# Define features and target
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Add constant for intercept
X = sm.add_constant(X)

# Fit the OLS model
try:
    model = sm.OLS(y, X).fit()
except ValueError as e:
    st.error(f"Error fitting model: {e}")
    st.stop()

# Prediction function
def predict_yield(area, item, year, rain, pesticides, temp):
    input_df = pd.DataFrame({
        'Year': [year],
        'average_rain_fall_mm_per_year': [rain],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [temp]
    })
    for col in dummies_area.columns:
        input_df[col] = 1 if col == f'Area_{area}' else 0
    for col in dummies_item.columns:
        input_df[col] = 1 if col == f'Item_{item}' else 0
    input_df = sm.add_constant(input_df, has_constant='add')
    predicted_yield = model.predict(input_df)[0]
    return predicted_yield

# Suitability function
def is_suitable(area, item, predicted_yield):
    hist = df_original[(df_original['Area'] == area) & (df_original['Item'] == item)]
    if hist.empty:
        return "No historical data for this area and item"
    mean_yield = hist['hg/ha_yield'].mean()
    return "Suitable" if predicted_yield >= mean_yield else "Not Suitable"

# Function to predict for upcoming year
def predict_for_upcoming_year(area, item, future_year):
    hist = df_original[(df_original['Area'] == area) & (df_original['Item'] == item)]
    if hist.empty:
        return {"error": "No historical data for this area and item"}
    avg_rain = hist['average_rain_fall_mm_per_year'].mean()
    avg_pest = hist['pesticides_tonnes'].mean()
    avg_temp = hist['avg_temp'].mean()
    predicted_yield = predict_yield(area, item, future_year, avg_rain, avg_pest, avg_temp)
    suitability = is_suitable(area, item, predicted_yield)
    return {
        "predicted_yield": predicted_yield,
        "suitability": suitability,
        "avg_rain": avg_rain,
        "avg_pesticides": avg_pest,
        "avg_temp": avg_temp
    }

# Streamlit App Interface
st.title("Crop Yield Prediction App")
st.write("Predict crop yield and suitability for a specified season or upcoming year.")

# Input form
st.header("Input Parameters for Prediction")
area = st.selectbox("Select Area", df_original['Area'].unique())
item = st.selectbox("Select Crop", df_original['Item'].unique())
year = st.number_input("Enter Year", min_value=1990, max_value=2050, value=2025)
rain = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1000.0)
pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=100.0)
temp = st.number_input("Average Temperature (°C)", min_value=0.0, value=20.0)

# Predict button
if st.button("Predict Yield"):
    predicted_yield = predict_yield(area, item, year, rain, pesticides, temp)
    suitability = is_suitable(area, item, predicted_yield)
    st.subheader("Prediction Results")
    st.write(f"**Predicted Yield**: {predicted_yield:.2f} hg/ha")
    st.write(f"**Suitability**: {suitability}")

# Future prediction
st.header("Predict for Upcoming Year")
future_year = st.number_input("Enter Future Year", min_value=2025, max_value=2050, value=2026)
if st.button("Predict for Future Year"):
    future_pred = predict_for_upcoming_year(area, item, future_year)
    st.subheader(f"Prediction for {future_year}")
    if "error" in future_pred:
        st.error(future_pred["error"])
    else:
        st.write(f"**Predicted Yield**: {future_pred['predicted_yield']:.2f} hg/ha")
        st.write(f"**Suitability**: {future_pred['suitability']}")
        st.write(f"**Based on Historical Averages**:")
        st.write(f"- Rainfall: {future_pred['avg_rain']:.2f} mm/year")
        st.write(f"- Pesticides: {future_pred['avg_pesticides']:.2f} tonnes")
        st.write(f"- Temperature: {future_pred['avg_temp']:.2f} °C")

# Historical context
st.header("Historical Context")
hist = df_original[(df_original['Area'] == area) & (df_original['Item'] == item)]
if not hist.empty:
    mean_yield = hist['hg/ha_yield'].mean()
    st.write(f"**Historical Mean Yield for {item} in {area}**: {mean_yield:.2f} hg/ha")
else:
    st.write("No historical data available for this area and crop.")