# Predictor Laboral — RR.HH.

App web con Flask que predice si un candidato será contratado, usando un modelo Random Forest entrenado sobre datos de colocación universitaria.

---

## Requisitos

- Python 3.8+
- pip

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/guia7.git
cd guia7

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install flask numpy scikit-learn
```

---

## Estructura del proyecto

```
├── app.py                  # Servidor Flask
├── model_rf.pkl            # Modelo Random Forest entrenado
├── scaler.pkl              # Scaler (StandardScaler)
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
| N° | Número de registro |
| Género | M / F |
| Nota SSC | Secundaria (escala 0–20) |
| Nota HSC | Pre-universitario / CEPRE / IB (0–20) |
| Nota Grado | Universitaria (0–20) |
| Experiencia laboral | Sí / No |
| Nota E-Test | Prueba de empleabilidad (0–20) |
| Especialización | Mkt&Fin / Mkt&HR |
| Nota MBA | Posgrado (0–20) |

Presionar **Predecir** → resultado: **Contratado** o **No Contratado**.

---

## Dependencias

```
flask==2.2.5
numpy==1.24.4
scikit-learn==1.5.0
```

Para generar `requirements.txt`:

```bash
pip freeze > requirements.txt
```
