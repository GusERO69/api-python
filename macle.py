from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import pymysql

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route('/predict/<int:user_id>', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def predict(user_id):
    print('Recibida solicitud de predicción para el usuario:', user_id)
    # Conectar a la base de datos MySQL
    try:
        conn = pymysql.connect(
            host='34.151.233.27',
            user='CEG4',
            password="'sFk)z/lm1l7nD2;",
            database='Grupo4'
        )
        print("Conectado a la base de datos MySQL")
    except pymysql.MySQLError as e:
        return f"Error al conectar a la base de datos: {e}", 500

    cursor = conn.cursor()
    query = f"SELECT power, timestamp FROM Lectura WHERE user_id = {user_id}"
    cursor.execute(query)

    results = cursor.fetchall()
    
    if not results:
        conn.close()
        return jsonify({'error': f"No existe el usuario con ID: {user_id}"}), 404

    df = pd.DataFrame(results, columns=['power', 'timestamp'])

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day

    X = df[['day', 'month', 'year']]
    y = df['power']

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicción con nuevos datos
    new_data = pd.DataFrame([[19, 7, 2024]], columns=['day', 'month', 'year'])
    prediction = model.predict(new_data)
    
    conn.close()
    return jsonify(prediction=float(prediction))

if __name__ == '__main__':
    app.run(port=5000)
