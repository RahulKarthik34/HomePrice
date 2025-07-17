# 🏡 Home Price Predictor (Streamlit + XGBoost)

An interactive web application that predicts housing prices based on various features such as bedrooms, bathrooms, area, ZIP code, year built, and more. Built using Python, Streamlit, and XGBoost, and powered by real estate data from King County, USA.

---


## 📁 Project Structure

```

home-price-predictor/
├── home\_price\_streamlit.py     # Main Streamlit app
├── kc\_house\_data.csv           # Dataset (King County Housing Data)
├── README.md                   # Project documentation

````

---

## 📊 Features

- 🔢 Predict house prices using **XGBoost Regressor**
- 🧮 Dynamic sliders and dropdowns for user input
- 📍 ZIP code-based prediction
- 📈 Interactive price trend charts (Year Built & Renovated)
- ⬇️ Downloadable chart data as CSV

---

## 🛠️ Technologies Used

- Python 3.x  
- [Streamlit](https://streamlit.io/) – Web app framework  
- [XGBoost](https://xgboost.readthedocs.io/) – Machine Learning model  
- pandas, numpy, scikit-learn – Data processing  
- altair – Interactive visualizations

---

## 🧪 How It Works

- Loads housing data from `kc_house_data.csv`
- Encodes ZIP codes for machine learning input
- Trains an **XGBoost Regressor** on selected numerical features
- Accepts user input through Streamlit widgets
- Predicts price in **USD**, converted to **INR** (₹) using a fixed rate
- Displays price trends by **year built** or **renovation year**

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/home-price-predictor.git
cd home-price-predictor
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install streamlit pandas numpy xgboost scikit-learn altair
```

### 3. Run the Streamlit App

```bash
streamlit run home_price_streamlit.py
```

---

## 🗃️ Dataset Info

* Dataset: `kc_house_data.csv`
* Source: [Kaggle - King County House Data](https://www.kaggle.com/harlfoxem/housesalesprediction)
* Features used:

  * `bedrooms`, `bathrooms`, `sqft_living`, `floors`
  * `condition`, `grade`, `view`
  * `zipcode`, `yr_built`, `yr_renovated`

---

## 🧩 Possible Enhancements

* 📍 Map view of ZIP code locations
* 📉 Feature importance visualization
* 📤 Upload your own dataset for predictions
* 📊 Export charts as images
* 🌐 Deploy to Streamlit Cloud or HuggingFace Spaces

---


## 🙌 Acknowledgments

* King County House Sales Dataset – Kaggle
* Streamlit for making data apps easy and fast
* XGBoost for robust regression modeling

---

