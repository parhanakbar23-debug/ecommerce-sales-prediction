from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = joblib.load('../models/linear_regression.pkl')

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    '../data/raw/all_months_clean.csv',
    sep=';'
)

# =========================
# PREPROCESS DATE
# =========================
df['Waktu Pesanan Dibuat'] = pd.to_datetime(
    df['Waktu Pesanan Dibuat'],
    errors='coerce'
)

# BUAT KOLOM MONTH
df['month'] = df[
    'Waktu Pesanan Dibuat'
].dt.month

# =========================
# DASHBOARD DATA
# =========================

total_sales = int(
    df['Total Pembayaran'].sum()
)

total_transactions = len(df)

avg_sales = int(
    df['Total Pembayaran'].mean()
)

# SALES PER BULAN
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

        months=monthly_sales[
            'month'
        ].tolist(),

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
            request.form[
                'num_product_categories'
            ]
        )

        ongkir = float(
            request.form['ongkir']
        )

        # TOTAL 18 FEATURES
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

        prediction = model.predict(
            features
        )[0]

    return render_template(

        'predict.html',

        prediction=prediction
    )

# =========================
# COMPARISON
# =========================


@app.route('/comparison')
def comparison():

    return render_template(
        'comparison.html'
    )


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
