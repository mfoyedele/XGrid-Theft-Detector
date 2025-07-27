
from flask import Flask, render_template, request
import pickle
import numpy as np
filename = 'xgrid-theft-prediction-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        energy_m = float(request.form['energy_mean'])
        energy_st = float(request.form['energy_std'])
        energy_ma = float(request.form['energy_max'])
        energy_mi = float(request.form['energy_min'])
        energy_rang = float(request.form['energy_range'])
        energy_freq_m = float(request.form['energy_freq_mean'])
        energy_freq_st = float(request.form['energy_freq_std'])
        energy_freq_ma = float(request.form['energy_freq_max'])
        energy_freq_mi = float(request.form['energy_freq_min'])
        
        data = np.array([[energy_m, energy_st, energy_ma, energy_mi, energy_rang, energy_freq_m,
                          energy_freq_st, energy_freq_ma, energy_freq_mi]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)