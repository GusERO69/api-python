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
            host='localhost',
            user='root',
            password="123456",
            database='grupo4'
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
    
    # Generar rango de fechas minuto a minuto en lugar de por día
    today = datetime.now().date()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=6)
    total_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    csv_files = []
    for date in pd.date_range(start=start_date, end=end_date):
        csv_file = generate_csv_for_day(user_id, date.date())
        if csv_file:
            csv_files.append(csv_file)
    
    if not csv_files:
        return jsonify({'error': f"No existen registros para el usuario con ID: {user_id} en los últimos 7 días desde {start_date} hasta {end_date}"}), 404

    # Leer los datos del CSV
    dfs = []
    real_data = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        dfs.append(df)
        
        hourly_avg = df.groupby(df['timestamp'].dt.floor('H'))['power'].mean()  # Agrupar por hora
        for ts, avg_power in hourly_avg.items():
            real_data[ts.strftime('%Y-%m-%d %H:%M:%S')] = float(avg_power)
    
    if not dfs:
        return jsonify({'error': f"Error al leer los datos de los CSV para el usuario con ID: {user_id}"}), 500
    
    df = pd.concat(dfs)
    
    mode_power = df['power'].mode().iloc[0]
    print(f"La moda de los datos de 'power' es: {mode_power}")
    
    X = df[['day', 'month', 'year', 'hour']]
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
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Predicción minuto a minuto
    predictions = []
    for date in total_dates:
        new_data = pd.DataFrame([[date.day, date.month, date.year, date.hour]], columns=['day', 'month', 'year', 'hour'])
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        predictions.append(prediction.flatten()[0])
        print(f"Predicción para el usuario con ID: {user_id} para la hora {date}: {prediction}")
    
    predictions_json = []
    for date, pred in zip(total_dates, predictions):
        pred = float(pred)
        mayor = 1 if pred > mode_power else 0
        probabilidad = max(0, 1 - abs(pred - mode_power) / abs(mode_power)) * 100
        error = 100 - probabilidad
        predictions_json.append({
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': pred,
            'mayor': mayor,
            'probabilidad': probabilidad,
            'error': error,
            'real': real_data.get(date.strftime('%Y-%m-%d %H:%M:%S'), None)
        })
        
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password="123456",
            database='grupo4'
        )
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO results (user_id, prediction, date_week, higher, probability, error) 
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        print(predictions_json)
        
        for prediction in predictions_json:
            date = prediction['date']
            pred = prediction['prediction']
            higher = prediction['mayor']
            proba = prediction['probabilidad']
            err = prediction['error']
            
            cursor.execute(insert_query, (user_id, pred, date, higher, proba, err))
        conn.commit()
        print(f"Predicciones para el usuario con ID: {user_id} insertadas correctamente en la base de datos.")
    except pymysql.MySQLError as e:
        print(f"Error al insertar las predicciones en la base de datos: {e}")
    finally:
        conn.close()
    
    # Graficar datos reales y predicciones
    plot_predictions(df, predictions, scaler, user_id, total_dates)
    
    return jsonify(predictions=predictions_json)

def plot_predictions(df, predictions, scaler, user_id, total_dates):
    # Obtener datos reales para el gráfico
    real_data = df.set_index('timestamp')['power']
    real_data = real_data.resample('H').mean()  # Promediar datos por minuto
    
    # Preparar datos de predicción para el gráfico
    pred_data = pd.Series(predictions, index=total_dates)
    
    # Graficar datos reales y predicciones
    plt.figure(figsize=(12, 6))
    plt.plot(real_data.index, real_data.values, label='Datos reales')
    plt.plot(pred_data.index, pred_data.values, marker='o', linestyle='-', color='r', label='Predicción')
    plt.title(f'Datos reales vs Predicción para usuario {user_id}')
    plt.xlabel('Fecha')
    plt.ylabel('Potencia')
    plt.legend()
    plt.grid(True)
    
    for date, real_value in zip(real_data.index, real_data.values):
        plt.text(date, real_value + 0.4, f'{real_value:.2f}', ha='center', va='bottom', fontsize=8, color='blue')
    
    for date, pred_value in zip(pred_data.index, pred_data.values):
        plt.text(date, pred_value + 0.1, f'{pred_value:.2f}', ha='center', va='bottom', fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Guardar gráfico como imagen
    plt.savefig(f'./{user_id}/weekly_prediction_plot.png')
    
    # Mostrar gráfico en la aplicación Flask (opcional)
    # plt.show()

if __name__ == '__main__':
    app.run(port=5000)
