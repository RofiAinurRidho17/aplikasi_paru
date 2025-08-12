from flask import Flask, request, jsonify, render_template
import joblib
# import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

model_path = 'model/best_model.pkl'
model = joblib.load(model_path)

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, 'user_predictions.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediksi')
def predic_page():
    return render_template('predic.html')

@app.route('/visualisasi')
def visualisasi_page():
    return render_template('visualisasi.html')

@app.route('/tentang')
def tentang_page():
    return render_template('tentang.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        data['usia_muda'] = 1 - data['usia_tua']
        data['jenis_kelamin_pria'] = 1 - data['jenis_kelamin_wanita']
        data['merokok_aktif'] = 1 - data['merokok_pasif']
        data['bekerja_tidak'] = 1 - data['bekerja_ya']
        data['rumah_tangga_tidak'] = 1 - data['rumah_tangga_ya']
        data['aktivitas_begadang_tidak'] = 1 - data['aktivitas_begadang_ya']
        data['aktivitas_olahraga_jarang'] = 1 - data['aktivitas_olahraga_sering']
        data['asuransi_ada'] = 1 - data['asuransi_tidak']
        data['penyakit_bawaan_ada'] = 1 - data['penyakit_bawaan_tidak']

        feature_order = [
            'usia_muda',
            'usia_tua',
            'jenis_kelamin_pria',
            'jenis_kelamin_wanita',
            'merokok_aktif',
            'merokok_pasif',
            'bekerja_tidak',
            'bekerja_ya',
            'rumah_tangga_tidak',
            'rumah_tangga_ya',
            'aktivitas_begadang_tidak',
            'aktivitas_begadang_ya',
            'aktivitas_olahraga_jarang',
            'aktivitas_olahraga_sering',
            'asuransi_ada',
            'asuransi_tidak',
            'penyakit_bawaan_ada',
            'penyakit_bawaan_tidak',
        ]

        input_values = [data[feature] for feature in feature_order]
        input_df = pd.DataFrame([input_values], columns=feature_order)

        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability >= 0.40)

        row = {**data,
       "prediction": prediction,
       "probability": probability,
       "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        df_row = pd.DataFrame([row])

        if os.path.exists(CSV_PATH):
            df_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            df_row.to_csv(CSV_PATH, mode='w', header=True, index=False)

        return jsonify({
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})


@app.route('/visualisasi-data')
def visualisasi_data():
    try:
        df = pd.read_csv(CSV_PATH)
        total_berisiko = df['prediction'].sum()
        total_tidak_berisiko = len(df) - total_berisiko

        return jsonify({
            "berisiko": int(total_berisiko),
            "tidak_berisiko": int(total_tidak_berisiko)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/riwayat-prediksi')
def riwayat_prediksi():
    try:
        df = pd.read_csv(CSV_PATH)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        kolom_urutan = [
            'usia_tua', 'jenis_kelamin_wanita', 'merokok_pasif', 'bekerja_ya',
            'rumah_tangga_ya', 'aktivitas_begadang_ya', 'aktivitas_olahraga_sering',
            'asuransi_tidak', 'penyakit_bawaan_tidak', 'prediction', 'probability', 'timestamp'
        ]

        kolom_ada = [col for col in kolom_urutan if col in df.columns]
        df = df[kolom_ada]
        if 'probability' in df.columns:
            df['probability'] = (df['probability'] * 100).round(2).astype(str) + '%'

        df.insert(0, 'no', range(1, len(df) + 1))

        # df = df[[col for col in kolom_urutan if col in df.columns]]

        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)})

    
@app.route('/visualisasi-kategori/<kolom>')
def visualisasi_kategori(kolom):
    try:
        df = pd.read_csv(CSV_PATH)

        if kolom not in df.columns:
            return jsonify({"error": f"Kolom '{kolom}' tidak ditemukan dalam data."})

        mapping_dict = {
            'usia_tua': {0: 'Muda', 1: 'Tua'},
            'jenis_kelamin_wanita': {0: 'Pria', 1: 'Wanita'},
            'merokok_pasif': {0: 'Aktif', 1: 'Pasif'},
            'bekerja_ya': {0: 'Tidak', 1: 'Ya'},
            'rumah_tangga_ya': {0: 'Tidak', 1: 'Ya'},
            'aktivitas_begadang_ya': {0: 'Tidak', 1: 'Ya'},
            'aktivitas_olahraga_sering': {0: 'Jarang', 1: 'Sering'},
            'asuransi_tidak': {0: 'Ya', 1: 'Tidak'},
            'penyakit_bawaan_tidak': {0: 'Ya', 1: 'Tidak'}
        }

        if 'prediction' not in df.columns:
            return jsonify({"error": "Kolom prediction tidak ditemukan dalam data."})

        if kolom in mapping_dict:
            df[kolom] = df[kolom].map(mapping_dict[kolom])

        grouped = df.groupby([kolom, 'prediction']).size().unstack(fill_value=0)

        labels = grouped.index.astype(str).tolist()
        berisiko = grouped[1].tolist() if 1 in grouped.columns else [0] * len(labels)
        tidak_berisiko = grouped[0].tolist() if 0 in grouped.columns else [0] * len(labels)

        return jsonify({
            "labels": labels,
            "berisiko": berisiko,
            "tidak_berisiko": tidak_berisiko
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
