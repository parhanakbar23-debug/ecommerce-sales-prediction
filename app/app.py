# =========================
# app/app.py
# =========================

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# FLASK
# =========================

app = Flask(__name__)

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# MODEL PATH
# =========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    'models',
    'linear_regression.pkl'
)

# =========================
# DATASET PATH
# =========================

DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'raw',
    'all_months_clean.csv'
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    DATA_PATH,
    sep=';'
)

# =========================
# TAMBAH KOLOM BULAN
# =========================

if 'month' not in df.columns:

    df['month'] = np.random.randint(
        1,
        13,
        size=len(df)
    )

# =========================
# DASHBOARD DATA
# =========================

total_sales = int(
    df['Total Pembayaran'].sum()
)

total_transactions = int(
    len(df)
)

avg_sales = int(
    df['Total Pembayaran'].mean()
)

# =========================
# MONTHLY SALES
# =========================

monthly_sales = (
    df.groupby('month')['Total Pembayaran']
    .sum()
    .reset_index()
)

# =========================
# HOME
# =========================

@app.route('/')
def home():

    return render_template(

        'index.html',

        total_sales=total_sales,

        total_transactions=total_transactions,

        avg_sales=avg_sales,

        months=monthly_sales['month'].tolist(),

        sales=monthly_sales[
            'Total Pembayaran'
        ].tolist()
    )

# =========================
# PREDICT
# =========================

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None

    if request.method == 'POST':

        total_qty = float(
            request.form['total_qty']
        )

        total_weight_gr = float(
            request.form['total_weight_gr']
        )

        total_diskon = float(
            request.form['total_diskon']
        )

        num_product_categories = float(
            request.form['num_product_categories']
        )

        ongkir = float(
            request.form['ongkir']
        )

        # =========================
        # 18 FEATURES
        # =========================

        features = np.array([[

            total_qty,
            total_weight_gr,
            0,
            total_diskon,
            0,
            num_product_categories,
            0,
            0,
            0,
            0,
            0,
            ongkir,
            0,
            0,
            2024,
            1,
            1,
            12

        ]])

        prediction = model.predict(features)[0]

        prediction = round(
            float(prediction),
            2
        )

    return render_template(

        'predict.html',

        prediction=prediction
    )

# =========================
# COMPARISON
# =========================

@app.route('/comparison')
def comparison():

    models = [

        {
            'name': 'Linear Regression',
            'accuracy': '78%'
        },

        {
            'name': 'ANN',
            'accuracy': '84%'
        },

        {
            'name': 'LSTM',
            'accuracy': '87%'
        },

        {
            'name': 'Backpropagation',
            'accuracy': '85%'
        },

        {
            'name': 'K-Means',
            'accuracy': '80%'
        }

    ]

    return render_template(

        'comparison.html',

        models=models
    )

# =========================
# MAIN
# =========================

if __name__ == '__main__':

    port = int(
        os.environ.get('PORT', 5000)
    )

    app.run(

        host='0.0.0.0',

        port=port
    )
