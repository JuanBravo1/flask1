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
    <head><title>Clasificador de Iris</title></head>
    <body>
    <h1>Clasificador de Flores Iris 🌸</h1>
    <form method="POST">
        <label>Longitud del sépalo:</label>
        <input type="text" name="sepal_length" required><br><br>
        <label>Ancho del sépalo:</label>
        <input type="text" name="sepal_width" required><br><br>
        <label>Longitud del pétalo:</label>
        <input type="text" name="petal_length" required><br><br>
        <label>Ancho del pétalo:</label>
        <input type="text" name="petal_width" required><br><br>
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
            html += f"<h2>Predicción: {pred[0]}</h2>"
        except:
            html += "<h2 style='color:red;'>Error en los datos</h2>"

    html += "</body></html>"
    return html

if __name__ == '__main__':
    app.run(debug=True)
