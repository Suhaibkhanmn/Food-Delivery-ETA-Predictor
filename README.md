# Food Delivery ETA Predictor

This project predicts **how long a food delivery will take** based on things like distance, weather, traffic, preparation time, courier experience, and more.  
It’s built for **hyperlocal delivery apps** and can be adapted for other last-mile delivery scenarios.

---

## What This Project Does
Imagine you order food from a restaurant. This model looks at your order details and predicts **how many minutes it will take** for the delivery to reach you.  

Some key highlights:
- **Interactive web app** built with [Streamlit](https://streamlit.io)  
- A trained **Random Forest** machine learning model  
- Easy-to-read predictions, plus a **confidence range**  
- Clean, modular notebook workflow for exploration and model building

---

## Getting Started

### 1️⃣ Clone this repository

git clone https://github.com/your-username/food-delivery-eta.git
cd food-delivery-eta

### 2️⃣ Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac / Linux

###3️⃣ Install the dependencies

pip install --upgrade pip
pip install -r requirements.txt

###▶️ Run the Web App

streamlit run app.py

Then open your browser at:
http://localhost:8501

---

### 📊 Model Performance

Mean Absolute Error (MAE): ~6.7 minutes

Median Error: ~4.8 minutes

90% of predictions within ~14 minutes

This means that for most orders, the predicted time is within about 7 minutes of the actual time.


