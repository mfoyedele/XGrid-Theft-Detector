from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import numpy as np
import pandas as pd
from scipy.fft import fft
import joblib
import os
import zipfile
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load model and scaler
classifier = joblib.load('xgrid-theft-prediction-model.pkl')
scaler = joblib.load('scaler.pkl')


# Optional: extract features from raw time-series (not used in this version)
def extract_features(data):
    freq = np.abs(fft(data))[:len(data) // 2]
    return pd.Series({
        'energy_mean': data.mean(),
        'energy_std': data.std(),
        'energy_max': data.max(),
        'energy_min': data.min(),
        'energy_range': data.max() - data.min(),
        'energy_freq_mean': freq.mean(),
        'energy_freq_std': freq.std(),
        'energy_freq_max': freq.max(),
        'energy_freq_min': freq.min()
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Manual Input ---
    if all(k in request.form for k in [
        'energy_mean', 'energy_std', 'energy_max', 'energy_min', 'energy_range',
        'energy_freq_mean', 'energy_freq_std', 'energy_freq_max', 'energy_freq_min'
    ]):
        try:
            # Get form data
            data = np.array([[
                float(request.form['energy_mean']),
                float(request.form['energy_std']),
                float(request.form['energy_max']),
                float(request.form['energy_min']),
                float(request.form['energy_range']),
                float(request.form['energy_freq_mean']),
                float(request.form['energy_freq_std']),
                float(request.form['energy_freq_max']),
                float(request.form['energy_freq_min']),
            ]])

            scaled_data = scaler.transform(data)
            prediction = classifier.predict(scaled_data)[0]
            label = "Theft" if prediction == 1 else "Non-theft"

            return render_template("result.html", result=label)

        except Exception as e:
            return render_template("result.html", error=f"Invalid input: {str(e)}")

    # --- File Upload ---
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template("result.html", error="No file selected.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read file
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            else:
                return render_template("result.html", error="Unsupported file type. Upload .csv or .xlsx.")

            expected_columns = [
                'customer_id', 'energy_mean', 'energy_std', 'energy_max', 'energy_min',
                'energy_range', 'energy_freq_mean', 'energy_freq_std', 'energy_freq_max', 'energy_freq_min'
            ]
            if not all(col in df.columns for col in expected_columns):
                return render_template("result.html", error="Missing one or more required columns.")

            features = df[expected_columns[1:]]
            features_scaled = scaler.transform(features)
            preds = classifier.predict(features_scaled)
            df['Prediction'] = ['Theft' if p == 1 else 'Non-theft' for p in preds]

            # Save result to CSV
            result_filename = f"prediction_results_{timestamp}.csv"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            df.to_csv(result_path, index=False)

            # Create ZIP
            zip_filename = f"results_{timestamp}.zip"
            zip_path = os.path.join(app.config['RESULT_FOLDER'], zip_filename)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(result_path, arcname=result_filename)

            results = [f"{row.customer_id}: {row.Prediction}" for _, row in df.iterrows()]

            return render_template(
                "result.html",
                preview_table=df.head().to_html(classes='table table-striped', justify='left'),
                summary_table=df['Prediction'].value_counts().to_frame().T.to_html(classes='table table-bordered', justify='left'),
                batch_result=results,
                download_link=f"/download/{zip_filename}"
            )

        except Exception as e:
            return render_template("result.html", error=f"Error processing file: {str(e)}")

    return redirect(url_for('index'))


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

