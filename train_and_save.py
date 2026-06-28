import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def build_ae(latent_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(latent_dim, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation=None)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    print("1. 데이터 로드 및 전처리 중...")
    df = pd.read_csv('c:/AI_1team/traffic_features.csv')
    features = ['car_count', 'bus_count', 'truck_count', 'density']
    X = df[features].values.astype(float)
    
    # 노트북과 동일한 방식으로 임의 이상치 주입
    y_true = np.zeros(len(X))
    anomaly_idx = int(len(X) * 0.9)
    X[anomaly_idx:, 0] *= 3.5  
    X[anomaly_idx:, 3] *= 4.0  
    y_true[anomaly_idx:] = 1
    
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("2. Autoencoder 모델 학습 중...")
    ae = build_ae(16)
    ae.fit(X_train_s, X_train_s, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    
    print("3. Threshold 계산 중...")
    mse = np.mean(np.square(X_test_s - ae.predict(X_test_s)), axis=1)
    threshold = float(np.percentile(mse, 80))
    print(f" -> 계산된 이상탐지 Threshold: {threshold:.4f}")
    
    print("4. 모델, 스케일러, 임계값 저장 중...")
    os.makedirs('c:/AI_1team/models', exist_ok=True)
    
    # 모델 저장 (Keras 포맷)
    ae.save('c:/AI_1team/models/autoencoder.keras')
    
    # 스케일러 저장
    joblib.dump(scaler, 'c:/AI_1team/models/scaler.pkl')
    
    # Threshold 저장
    with open('c:/AI_1team/models/config.json', 'w') as f:
        json.dump({'threshold': threshold}, f)
        
    print("완료! 모든 파일이 c:/AI_1team/models 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()
