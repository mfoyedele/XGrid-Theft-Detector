
# ⚡ XGrid Theft Detector

**Smart AI-powered Electricity Theft Detection Web App**

XGrid Theft Detector is an open-source AI solution that helps identify electricity theft using machine learning models trained on historical energy usage patterns. It enables users to either manually input data or upload datasets (CSV/Excel) and get instant predictions.

---
![alt text](https://github.com/mfoyedele/XGrid-Theft-Detector/blob/main/assets/xgrid_homepage.png?raw=true)
![alt text](https://github.com/mfoyedele/XGrid-Theft-Detector/blob/main/assets/xgrid_prediction.png?raw=true)
---
## Features

- Predict **Theft vs Non-Theft** based on 9 extracted features
- Upload `.csv` or `.xlsx` files for **batch predictions**
- Download predictions in a **ZIP archive**
- View summaries and top predictions on the web
- Powered by **XGBoost** with `StandardScaler` preprocessing

---

## How It Works

1. **Extract Features** from energy readings (e.g., mean, std, max, FFT stats)
2. **Preprocess** using StandardScaler
3. **Predict** using a trained XGBoost classifier
4. **Display & Export** results (ZIP & CSV)

---

## Project Structure

```
XGrid/
│
├── app.py                    # Flask web server
├── model.py                  # ML training and evaluation
├── data                      # ML selected extract features jupyter notebook and SGCC dataset
├── assets                    # App images
├── csv_test_files            # Contains csv files for testing
├── excel_test_files          # Contains excel files for testing
├── notebooks                 # Contains ML jupyter notebooks
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/                  # Uploaded files
├── results/                  # Prediction output files
├── xgrid-theft-prediction-model.pkl  # Trained model
├── scaler.pkl                # Fitted scaler
├── README.md                 # This file
```

---

## Requirements

- Python 3.7+
- Flask
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Joblib
- OpenPyXL

Install using:

```bash
pip install -r requirements.txt
```

---

## Running the App Locally

```bash
python app.py
```

Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Training Your Own Model

If you want to retrain:

```bash
python model.py
```

It will generate:
- `xgrid-theft-prediction-model.pkl`
- `scaler.pkl`

---

## Inputs Required

Whether through form or file, you need the following 9 features:

- `energy_mean`
- `energy_std`
- `energy_max`
- `energy_min`
- `energy_range`
- `energy_freq_mean`
- `energy_freq_std`
- `energy_freq_max`
- `energy_freq_min`

> For file upload: add a `customer_id` column too.

---

## 📄 License

MIT License

---

## Contributors

- Musa Funso Oyedele – [@mfoyedele](https://github.com/mfoyedele)
- Open to contributors!
