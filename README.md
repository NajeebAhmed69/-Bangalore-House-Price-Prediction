```markdown
# 🏡 Bengaluru House Price Prediction

An end-to-end Machine Learning regression project that predicts residential real estate prices in Bengaluru (Bangalore), India, based on location, square footage, bedrooms (BHK), and bathrooms. Built with **Scikit-Learn**, **Pandas**, and **Streamlit**.

---

## 📌 Project Overview

Real estate valuation in rapidly expanding metropolitan areas like Bengaluru is influenced by geographic micro-markets, property dimensions, and structural configurations. This project addresses messy, real-world real estate data by engineering custom data-cleaning pipelines, mitigating high-cardinality location features, removing domain-specific outliers, and serving predictions through an interactive Streamlit UI.

---

## 📊 Dataset Highlights

- **Source:** Bengaluru House Price Dataset (`Bengaluru_House_Data.csv`)
- **Total Records:** 13,320 listings across 9 attributes
- **Target Variable:** `price` (in Indian Lakhs, where 1 Lakh = ₹100,000 INR)
- **Features Handled:**
  - `location`: Over 1,300 unique neighborhoods (reduced via frequency bucketing)
  - `total_sqft`: Square footage (handled numeric strings, ranges, and area conversions)
  - `size`: Extracted integer bedroom counts (`BHK`)
  - `bath`: Number of bathrooms

---

## ⚙️ Data Cleaning & Feature Engineering

1. **Feature Reduction & Handling Nulls:**
   - Dropped low-information or high-null features (`society`, `availability`, `area_type`, `balcony`).
   - Imputed and cleaned missing rows in core attributes (`location`, `sqft`, `bath`, `bhk`).
2. **Parsing Square Footage:**
   - Converted ranged values (e.g., `2100 - 2850`) into numeric means.
   - Stripped non-standard metric strings to maintain consistent numeric square footage.
3. **Location Dimensionality Reduction:**
   - Locations with 10 or fewer listings were aggregated into an `other` category, reducing dimensionality from 1,300+ categories to ~240 prominent localities.
4. **Domain-Specific Outlier Removal:**
   - **Square Footage per Bedroom:** Filtered out unrealistic configurations where $\text{sqft} / \text{BHK} < 300$.
   - **Price per Square Foot Outliers:** Removed extreme price anomalies beyond 1 standard deviation per locality ($\mu \pm 1\sigma$).
   - **Bathroom Anomalies:** Filtered properties where $\text{bath} \ge \text{BHK} + 2$.

---

## 🧠 Model Pipeline & Evaluation

- **Preprocessing:** Scikit-Learn `ColumnTransformer` applying `OneHotEncoder(drop='first', handle_unknown='ignore')` on `location` and passing numeric features through.
- **Model:** `LinearRegression` pipeline.
- **Key Metrics:**
  - **$R^2$ Score:** $\approx 0.77 - 0.84$
  - **MAE:** $\approx \text{₹19.6 Lakhs}$

---

## 🗂️ Project Structure

```text
├── Bengaluru_House_Data.csv      # Raw real estate dataset
├── House_prediction.py                # Cleaning, outlier treatment, training & artifact export
├── app.py                        # Streamlit web application
├── bangalore_house_model.pkl     # Exported Scikit-Learn pipeline
├── locations.pkl                 # Pickled list of supported localities
└── README.md                     # Project documentation

```

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone [https://github.com/NajeebAhmed69/-Bangalore-House-Price-Prediction.git](https://github.com/NajeebAhmed69/-Bangalore-House-Price-Prediction.git)
cd bengaluru-house-price-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


```

### 2. Train the Model & Export Pickles

```bash
python House_prediction.py

```

### 3. Run the Streamlit Dashboard

```bash
streamlit run app.py

```

Open `http://localhost:8501` in your browser to interact with the valuation app.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, Linear Models)
* **Model Serialization:** Pickle
* **Web Interface:** Streamlit

```

```