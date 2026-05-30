# Predictor Laboral — RR.HH.

App web con Flask que predice si un candidato será contratado, usando un modelo Random Forest entrenado sobre datos de colocación universitaria.

Incluye probabilidad de contratación con gauge visual, explicación por candidato usando SHAP (Shapley Additive Explanations) y advertencia sobre sesgos del modelo.

---

## Requisitos

- Python 3.8+
- pip

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Joaquin2705/guia7.git
cd guia7

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install flask numpy scikit-learn shap
```

---

## Estructura del proyecto

```
├── app.py                  # Servidor Flask
├── model_rf.pkl            # Modelo Random Forest entrenado
├── scaler.pkl              # Scaler (MinMaxScaler)
├── Templates/
│   └── index.html          # Interfaz web
└── [Student View] Recursos_Humanos.ipynb   # Notebook de entrenamiento
```

---

## Correr la app

```bash
python app.py
```

Abrir en el browser: [http://127.0.0.1:5001](http://127.0.0.1:5001)

---

## Uso

Ingresar los datos del candidato en el formulario (notas en escala vigesimal 0–20):

| Campo | Descripción |
|---|---|
| Género | M / F |
| Experiencia laboral | Sí / No |
| Secundaria | Promedio general del colegio (0–20) |
| Pre-universitario | CEPRE, IB u otro programa (0–20) |
| Pregrado | Promedio universitario (0–20) |
| MBA | Promedio del MBA (0–20) |
| Examen de aptitudes | Test de competencias laborales (0–20) |
| Especialización | Marketing y Finanzas / Marketing y RR.HH. |

Presionar **Predecir empleabilidad** para obtener el resultado.

---

## Resultado

- **Resultado**: Contratado o No Contratado
- **Probabilidad**: gauge visual con el porcentaje exacto de probabilidad de contratación
- **Factores clave**: los 8 features ordenados por impacto real en la predicción de ese candidato, calculados con SHAP. Cada factor muestra cuántos puntos porcentuales subió o bajó la probabilidad
- **Probabilidad base**: punto de partida fijo del modelo (promedio histórico del dataset de entrenamiento), igual para todos los candidatos

---

## Cómo funciona SHAP

El modelo parte de una probabilidad base (promedio histórico del training). Por cada candidato, SHAP calcula cuánto subió o bajó esa probabilidad por culpa de cada feature, recorriendo los árboles del Random Forest. La suma de todos los valores SHAP más la base es igual a la probabilidad final mostrada en el gauge.

---

## Dependencias principales

```
flask
numpy
scikit-learn
shap
```

Para generar `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## Nota sobre sesgos

El modelo fue entrenado con datos históricos de contratación. Los factores mostrados reflejan patrones del pasado, no criterios que deban usarse como regla. Se recomienda usar esta herramienta como apoyo informativo, no como decisión final.
