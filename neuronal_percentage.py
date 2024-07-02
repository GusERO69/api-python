from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import pymysql
from datetime import datetime, timedelta
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    
    last_timestamp = df['timestamp'].max()
    last_time = pd.to_datetime(last_timestamp)
    if last_time.time() >= datetime.strptime("23:59:00", "%H:%M:%S").time():
        print(f"El archivo CSV para el usuario con ID: {user_id} en la fecha: {date} ha completado los registros del día.")
    else:
        print(f"El archivo CSV para el usuario con ID: {user_id} en la fecha: {date} no ha completado todos los registros del día.")
    
    conn.close()
    return csv_file_path

@app.route('/predict/<int:user_id>', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def predict(user_id):
    print('Recibida solicitud de predicción para el usuario:', user_id)
    
    # Generar archivo CSV para la semana actual
    today = datetime.now().date()
    start_date = today
    end_date = start_date + timedelta(days=7)  # Predicción para la siguiente semana
    dates = pd.date_range(start=start_date, end=end_date)
    
    csv_files = []
    for date in dates:
        csv_file = generate_csv_for_day(user_id, date.date())
        if csv_file:
            csv_files.append(csv_file)
    
    if not csv_files:
        return jsonify({'error': f"No existen registros para el usuario con ID: {user_id} en la semana del {start_date} al {end_date}"}), 404

    # Leer los datos del CSV
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        dfs.append(df)
    
    if not dfs:
        return jsonify({'error': f"Error al leer los datos de los CSV para el usuario con ID: {user_id}"}), 500
    
    df = pd.concat(dfs)
    
    mode_power = df['power'].mode().iloc[0]
    print(f"La moda de los datos de 'power' es: {mode_power}")
    
    X = df[['day', 'month', 'year']]
    y = df['power']
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Definir modelo de red neuronal
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Compilar modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrenar modelo
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    
    # Predicción para la siguiente semana
    next_week = start_date + timedelta(days=7)
    next_week_dates = pd.date_range(start=start_date, end=next_week)
    predictions = []
    
    for date in next_week_dates:
        new_data = pd.DataFrame([[date.day, date.month, date.year]], columns=['day', 'month', 'year'])
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        predictions.append(prediction.flatten()[0])
        print(f"Predicción para el usuario con ID: {user_id} para el día {date}: {prediction}")
        
        predictions_json = []
        for date, pred in zip(next_week_dates, predictions):
            pred = float(pred)
            mayor = 1 if pred > mode_power else 0
            probabilidad = max(0, 1 - abs(pred - mode_power) / abs(mode_power)) * 100
            error = 100 - probabilidad
            predictions_json.append({
                'date': date.strftime('%Y-%m-%d'),
                'prediction': pred,
                'mayor': mayor,
                'probabilidad': probabilidad,
                'error': error
            })
        
        # try:
        #     conn = pymysql.connect(
        #         host='34.151.233.27',
        #         user='CEG4',
        #         password="'sFk)z/lm1l7nD2;",
        #         database='Grupo4'
        #     )
        #     cursor = conn.cursor()
        #     insert_query = """
        #     INSERT INTO results (user_id, prediction) 
        #     VALUES (%s, %s)
        #     """
        #     cursor.execute(insert_query, (user_id, prediction[0][0]))
        #     conn.commit()
        #     print(f"Predicción para el usuario con ID: {user_id} insertada correctamente en la base de datos.")
        # except pymysql.MySQLError as e:
        #     print(f"Error al insertar la predicción en la base de datos: {e}")
        # finally:
        #     conn.close()
    
    # Preparar respuesta JSON con las predicciones
    # predictions_json = [{'date': date.strftime('%Y-%m-%d'), 'prediction': float(pred), 'mayor': 1 if float(pred) > mode_power else 0} for date, pred in zip(next_week_dates, predictions)]
    
    # Graficar datos reales y predicciones (opcional)
    plot_predictions(df, model, scaler, user_id)
    
    return jsonify(predictions=predictions_json)

def plot_predictions(df, model, scaler, user_id):
    # Obtener datos reales para el gráfico
    real_data = df.set_index('timestamp')['power']
    real_data = real_data.resample('D').mean()  # Promediar datos diarios
    
    # Preparar datos de predicción para el gráfico
    start_date = datetime.now().date()
    next_week = start_date + timedelta(days=7)
    next_week_dates = pd.date_range(start=start_date, end=next_week)
    predictions = []
    
    for date in next_week_dates:
        new_data = pd.DataFrame([[date.day, date.month, date.year]], columns=['day', 'month', 'year'])
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        predictions.append(prediction.flatten()[0])
    
    pred_dates = pd.date_range(start=start_date, end=next_week)
    pred_data = pd.Series(predictions, index=pred_dates)
    
    # Graficar datos reales y predicciones
    plt.figure(figsize=(12, 6))
    plt.plot(real_data.index, real_data.values, label='Datos reales')
    plt.plot(pred_data.index, pred_data.values, marker='o', linestyle='-', color='r', label='Predicción')
    plt.title(f'Datos reales vs Predicción para usuario {user_id}')
    plt.xlabel('Fecha')
    plt.ylabel('Potencia')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Guardar gráfico como imagen (opcional)
    plt.savefig(f'./{user_id}/weekly_prediction_plot.png')
    
    # Mostrar gráfico en la aplicación Flask (opcional)
    # plt.show()  # Descomenta esta línea si deseas mostrar el gráfico en la aplicación

if __name__ == '__main__':
    app.run(port=5000)
