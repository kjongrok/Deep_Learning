import os
import cv2
import json
import joblib
import glob
import random
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

def main():
    print("1. 모델 및 설정 로드 중...")
    # 경로 설정
    yolo_model_path = r'c:\AI_1team\notebooks\yolov8n.pt'
    ae_model_path = r'c:\AI_1team\models\autoencoder.keras'
    scaler_path = r'c:\AI_1team\models\scaler.pkl'
    config_path = r'c:\AI_1team\models\config.json'

    # 모델 로드
    yolo_model = YOLO(yolo_model_path)
    ae_model = tf.keras.models.load_model(ae_model_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        threshold = config['threshold']
    
    print(f" -> 로드 완료 (이상탐지 임계값: {threshold:.4f})")

    # 샘플 이미지 하나 고르기
    print("2. 테스트용 이미지 선택 중...")
    search_path = r"c:\AI_1team\dataset\**\*.jpg"
    image_paths = glob.glob(search_path, recursive=True)
    if not image_paths:
        print("에러: dataset 폴더에서 이미지를 찾을 수 없습니다.")
        return
        
    sample_img_path = random.choice(image_paths)
    print(f" -> 선택된 이미지: {sample_img_path}")
    
    img = cv2.imread(sample_img_path)
    if img is None:
        print("이미지를 읽어오지 못했습니다.")
        return
        
    img_area = img.shape[0] * img.shape[1]
    
    print("3. YOLO를 통한 객체 탐지 수행 중...")
    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes
    
    car_count = 0
    bus_count = 0
    truck_count = 0
    total_bbox_area = 0.0
    
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id == 2:  # car
            car_count += 1
        elif cls_id == 5:  # bus
            bus_count += 1
        elif cls_id == 7:  # truck
            truck_count += 1
        else:
            continue
            
        # 바운딩 박스 넓이 계산 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w = x2 - x1
        h = y2 - y1
        total_bbox_area += (w * h)
        
    density = total_bbox_area / img_area if img_area > 0 else 0
    
    print(f" -> [추출된 특성] Car: {car_count}, Bus: {bus_count}, Truck: {truck_count}, Density: {density:.4f}")
    
    print("4. 이상 탐지(Anomaly Detection) 수행 중...")
    # 특성 배열 생성
    features = np.array([[car_count, bus_count, truck_count, density]], dtype=float)
    
    # 스케일링
    features_scaled = scaler.transform(features)
    
    # Autoencoder 예측 및 MSE 계산
    reconstructed = ae_model.predict(features_scaled, verbose=0)
    mse = np.mean(np.square(features_scaled - reconstructed), axis=1)[0]
    
    is_anomaly = mse > threshold
    
    print("="*50)
    print(f"분석 결과 MSE: {mse:.4f} (Threshold: {threshold:.4f})")
    if is_anomaly:
        print("🚨 [결과] 이상 상황(Anomaly)이 감지되었습니다! 교통 폭주 또는 사고 의심!")
    else:
        print("✅ [결과] 정상(Normal)적인 교통 흐름입니다.")
    print("="*50)

if __name__ == '__main__':
    # TensorFlow 경고 끄기
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
