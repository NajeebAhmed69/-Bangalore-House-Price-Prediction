import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Bengaluru House Price Estimator', page_icon='🏡', layout='centered'
)


# Load artifacts
@st.cache_resource
def load_artifacts():
  model = pickle.load(open('bangalore_house_model.pkl', 'rb'))
  locations = pickle.load(open('locations.pkl', 'rb'))
  return model, locations


try:
  model, locations = load_artifacts()
except FileNotFoundError:
  st.error(
      'Model files missing! Run `python train_model.py` first to generate'
      ' `bangalore_house_model.pkl` and `locations.pkl`.'
  )
  st.stop()

st.title('🏡 Bengaluru House Price Estimator')
st.markdown(
    'Estimate property market prices across Bangalore using trained regression'
    ' pipelines.'
)

# User Input Form
with st.form('prediction_form'):
  loc_choice = st.selectbox(
      'Select Locality / Area:', options=locations + ['other']
  )

  col1, col2 = st.columns(2)
  with col1:
    sqft_input = st.number_input(
        'Total Square Footage (sq ft):',
        min_value=300.0,
        max_value=25000.0,
        value=1200.0,
        step=50.0,
    )
    bhk_input = st.selectbox('BHK (Bedrooms):', [1, 2, 3, 4, 5], index=1)

  with col2:
    bath_input = st.selectbox('Number of Bathrooms:', [1, 2, 3, 4, 5], index=1)

  submit_btn = st.form_submit_button('Estimate Price', use_container_width=True)

if submit_btn:
  # Construct prediction DataFrame
  input_data = pd.DataFrame([{
      'location': loc_choice,
      'sqft': float(sqft_input),
      'bath': float(bath_input),
      'bhk': int(bhk_input),
  }])

  raw_prediction = model.predict(input_data)[0]
  predicted_lakhs = max(raw_prediction, 5.0)  # Bound against negative limits
  predicted_inr = predicted_lakhs * 100000

  st.divider()
  st.subheader('Valuation Estimate')
  st.success(f'### ₹ {predicted_lakhs:,.2f} Lakhs')
  st.caption(f'Estimated equivalent: ₹ {predicted_inr:,.0f} INR')