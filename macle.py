from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import pymysql
from datetime import datetime, timedelta
import schedule
import time

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

def generate_csv_for_day(user_id, date):
    try:
        conn = pymysql.connect(
            host='34.151.233.27',
            user='CEG4',
            password="'sFk)z/lm1l7nD2;",
            database='Grupo4'
        )
    except pymysql.MySQLError as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

    cursor = conn.cursor()
    query = f"""
    SELECT power, timestamp 
    FROM Lectura 
    WHERE user_id = {user_id} 
    AND DATE(timestamp) = '{date}'
    """
    cursor.execute(query)
    results = cursor.fetchall()
    
    if not results:
        conn.close()
        print(f"No existen registros para el usuario con ID: {user_id} en la fecha: {date}")
        return None

    df = pd.DataFrame(results, columns=['power', 'timestamp'])
    df.to_csv(f'{user_id}_{date}.csv', index=False)
    conn.close()
    return f'{user_id}_{date}.csv'

def scheduled_task():
    user_id = 1  # Cambia esto según sea necesario
    today = datetime.now().date()
    generate_csv_for_day(user_id, today)

schedule.every().day.at("00:00").do(scheduled_task)

@app.route('/predict/<int:user_id>', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def predict(user_id):
    print('Recibida solicitud de predicción para el usuario:', user_id)
    
    # Generar archivo CSV para el día actual
    today = datetime.now().date()
    csv_file = generate_csv_for_day(user_id, today)
    if not csv_file:
        return jsonify({'error': f"No existen registros para el usuario con ID: {user_id} en la fecha: {today}"}), 404

    # Leer los datos del CSV
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day

    X = df[['day', 'month', 'year']]
    y = df['power']

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicción para el siguiente día
    next_day = today + timedelta(days=1)
    new_data = pd.DataFrame([[next_day.day, next_day.month, next_day.year]], columns=['day', 'month', 'year'])
    prediction = model.predict(new_data)
    
    return jsonify(prediction=float(prediction))

if __name__ == '__main__':
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    import threading
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    
    app.run(port=5000)