from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load('modelo_svc.pkl')
escalador = joblib.load('escalador.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    html = """
    <html>
    <head>
        <title>Clasificador de Iris</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0f2f5;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            h1 {
                color: #333;
            }
            form {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                width: 300px;
            }
            label {
                margin-top: 10px;
                font-weight: bold;
            }
            input[type="text"] {
                padding: 8px;
                margin-top: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
                outline: none;
            }
            input[type="submit"] {
                margin-top: 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s ease;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                font-size: 18px;
                color: #333;
            }
            .error {
                color: red;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Clasificador de Flores Iris 🌸</h1>
        <form method="POST">
            <label>Longitud del sépalo:</label>
            <input type="text" name="sepal_length" required>
            <label>Ancho del sépalo:</label>
            <input type="text" name="sepal_width" required>
            <label>Longitud del pétalo:</label>
            <input type="text" name="petal_length" required>
            <label>Ancho del pétalo:</label>
            <input type="text" name="petal_width" required>
            <input type="submit" value="Predecir">
        </form>
    """

    if request.method == 'POST':
        try:
            datos = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            datos = np.array([datos])
            datos = escalador.transform(datos)
            pred = modelo.predict(datos)
            html += f"<div class='result'>🌼 <strong>Predicción:</strong> {pred[0]}</div>"
        except:
            html += "<div class='error'>❌ Error en los datos ingresados. Asegúrate de que todos sean numéricos.</div>"

    html += "</body></html>"
    return html

if __name__ == '__main__':
    app.run(debug=True)
