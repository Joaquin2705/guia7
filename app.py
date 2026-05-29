import pickle
import numpy as np
from flask import Flask, request, render_template

model = pickle.load(open('model_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def _parse_grade(form, name, label):
    try:
        val = float(form[name])
    except (ValueError, TypeError):
        raise ValueError(f"'{label}' debe ser un número.")
    if val < 0 or val > 20:
        raise ValueError(f"'{label}' debe estar entre 0 y 20 (escala vigesimal).")
    return val

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sl_no = 1

        gender = request.form.get('gender', '')
        if gender not in ('M', 'F'):
            raise ValueError("Género inválido.")

        workex = request.form.get('workex', '')
        if workex not in ('Yes', 'No'):
            raise ValueError("Experiencia laboral inválida.")

        specialisation = request.form.get('specialisation', '')
        if specialisation not in ('Mkt&Fin', 'Mkt&HR'):
            raise ValueError("Especialización inválida.")

        ssc_p    = _parse_grade(request.form, 'ssc_p',    'Secundaria')
        hsc_p    = _parse_grade(request.form, 'hsc_p',    'Pre-universitario / CEPRE / IB')
        degree_p = _parse_grade(request.form, 'degree_p', 'Pregrado')
        etest_p  = _parse_grade(request.form, 'etest_p',  'Examen de aptitudes')
        mba_p    = _parse_grade(request.form, 'mba_p',    'MBA')

    except ValueError as e:
        return render_template('index.html', error=str(e))

    gender_enc         = 1 if gender == 'M' else 0
    workex_enc         = 1 if workex == 'Yes' else 0
    specialisation_enc = 1 if specialisation == 'Mkt&Fin' else 0

    features = np.array([[sl_no, gender_enc, ssc_p * 5, hsc_p * 5, degree_p * 5,
                          workex_enc, etest_p * 5, specialisation_enc, mba_p * 5]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    result = 'Contratado' if prediction[0] == 1 else 'No Contratado'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
