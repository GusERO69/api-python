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
import os

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

    # Crear carpeta si no existe
    user_folder = f'./{user_id}'
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    df = pd.DataFrame(results, columns=['power', 'timestamp'])
    csv_file_path = os.path.join(user_folder, f'{user_id}_{date}.csv')
    df.to_csv(csv_file_path, index=False)
    conn.close()
    return csv_file_path

def insert_prediction_result(user_id, prediction):
    try:
        conn = pymysql.connect(
            host='34.151.233.27',
            user='CEG4',
            password="'sFk)z/lm1l7nD2;",
            database='Grupo4'
        )
    except pymysql.MySQLError as e:
        print(f"Error al conectar a la base de datos: {e}")
        return False

    cursor = conn.cursor()
    query = """
    INSERT INTO results (user_id, prediction) 
    VALUES (%s, %s)
    """
    try:
        cursor.execute(query, (user_id, prediction))
        conn.commit()
        conn.close()
        return True
    except pymysql.Error as e:
        print(f"Error al insertar el resultado de predicción: {e}")
        conn.rollback()
        conn.close()
        return False

def scheduled_task():
    user_id = 1  # Cambia esto según sea necesario
    today = datetime.now().date()
    csv_file = generate_csv_for_day(user_id, today)
    if not csv_file:
        print(f"No existen registros para el usuario con ID: {user_id} en la fecha: {today}")
        return
    
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
    prediction = model.predict(new_data)[0]

    # Insertar el resultado en la tabla results
    if insert_prediction_result(user_id, prediction):
        print(f"Predicción registrada en la tabla results para el usuario con ID {user_id}. Predicción: {prediction}")
    else:
        print(f"No se pudo registrar la predicción en la tabla results para el usuario con ID {user_id}.")
    
if __name__ == '__main__':
    schedule.every().day.at("00:00").do(scheduled_task)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
