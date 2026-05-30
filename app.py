import math
import pickle
import numpy as np
import shap
from flask import Flask, request, render_template

model   = pickle.load(open('model_rf.pkl', 'rb'))
scaler  = pickle.load(open('scaler.pkl', 'rb'))

# explainer se crea una sola vez al arrancar — es costoso inicializarlo cada request
explainer = shap.TreeExplainer(model)

app = Flask(__name__)

ARC_LEN = round(math.pi * 85, 1)  # semicircle r=85 ≈ 267.0

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

_FEATURE_LABELS = [
    None,                    # 0: sl_no (ignorar)
    'Genero',                # 1
    'Secundaria',            # 2
    'Pre-universitario',     # 3
    'Pregrado',              # 4
    'Experiencia laboral',   # 5
    'Examen de aptitudes',   # 6
    'Especializacion',       # 7
    'MBA',                   # 8
]

def _fmt_feature(feat_idx, raw_val):
    if feat_idx == 1:
        return 'Masculino' if raw_val == 1 else 'Femenino'
    if feat_idx == 5:
        return 'Con experiencia' if raw_val == 1 else 'Sin experiencia'
    if feat_idx == 7:
        return 'Mkt & Finanzas' if raw_val == 1 else 'Mkt & RR.HH.'
    return f"{raw_val / 5:.1f} / 20"

def _compute_factors(base_features, features_scaled):
    # SHAP funciona como un termometro que arranca en el promedio del training
    # (ej. 68% si historicamente 68 de cada 100 candidatos fueron contratados)
    # y sube o baja segun cada dato del candidato hasta llegar a la prediccion final.
    # Cada valor SHAP es cuanto subio o bajo ese "termometro" por culpa de ese feature.
    shap_vals = explainer.shap_values(features_scaled)

    # el formato de shap_values cambia segun la version de la libreria
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]    # version antigua: lista por clase -> tomamos clase 1 (Contratado)
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1] # version nueva 3D: (muestras, features, clases) -> primera muestra, clase 1
    else:
        sv = shap_vals[0]       # version nueva 2D: ya viene filtrado para clase positiva

    # sv[i] > 0 significa que ese feature subio la probabilidad de este candidato vs el promedio
    # sv[i] < 0 significa que la bajo
    contributions = []
    for i in range(1, 9):  # saltamos sl_no en idx 0, no aporta info real
        contrib = float(sv[i])
        contributions.append({
            'label':       _FEATURE_LABELS[i],
            'value':       _fmt_feature(i, base_features[0][i]),
            'contrib':     contrib,
            'abs_contrib': abs(contrib),
            # cuanto movio la probabilidad en puntos porcentuales reales
            'contrib_pct': round(contrib * 100, 1),
        })

    # ordenamos por magnitud — el que mas movio el termometro primero
    contributions.sort(key=lambda x: -x['abs_contrib'])

    # ancho de barra relativo al mayor para que se vea proporcional entre todos
    max_abs = contributions[0]['abs_contrib'] if contributions else 1
    for f in contributions:
        f['bar_pct']    = round(f['abs_contrib'] / max_abs * 100)
        f['direction']  = 'up' if f['contrib'] > 0 else 'down'
        f['contrib_str'] = ('+' if f['contrib'] > 0 else '') + str(f['contrib_pct']) + '%'

    return contributions

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

    base_features = np.array([[sl_no, gender_enc, ssc_p * 5, hsc_p * 5, degree_p * 5,
                               workex_enc, etest_p * 5, specialisation_enc, mba_p * 5]])

    # escalamos las features antes de pasarlas al modelo
    features_scaled = scaler.transform(base_features)

    # prediccion binaria: 1 = contratado, 0 = no contratado
    prediction      = model.predict(features_scaled)

    # predict_proba devuelve [prob_clase_0, prob_clase_1]
    # nos quedamos con proba[1] que es la probabilidad de ser contratado
    proba       = model.predict_proba(features_scaled)[0]
    probability = round(float(proba[1]) * 100, 1)

    result = 'Contratado' if prediction[0] == 1 else 'No Contratado'

    # color del gauge segun que tan alta es la probabilidad
    gauge_color  = '#34C759' if probability >= 70 else '#FF9500' if probability >= 40 else '#FF3B30'
    # offset controla cuanto arco SVG se rellena (0 = lleno, ARC_LEN = vacio)
    gauge_offset = round(ARC_LEN * (1 - probability / 100), 1)

    factors   = _compute_factors(base_features, features_scaled)
    base_prob = round(float(explainer.expected_value[1]) * 100, 1)

    form_vals = {
        'gender':         gender,
        'workex':         workex,
        'specialisation': specialisation,
        'ssc_p':          ssc_p,
        'hsc_p':          hsc_p,
        'degree_p':       degree_p,
        'etest_p':        etest_p,
        'mba_p':          mba_p,
    }

    return render_template('index.html',
                           result=result,
                           probability=probability,
                           gauge_color=gauge_color,
                           gauge_offset=gauge_offset,
                           arc_len=ARC_LEN,
                           factors=factors,
                           base_prob=base_prob,
                           form_vals=form_vals)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
