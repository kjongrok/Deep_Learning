import os
import cv2
import json
import joblib
import glob
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import winsound

def main():
    print("1. 모델 및 설정 로드 중...")
    yolo_model_path = r'notebooks\yolov8_nano.pt'
    ae_model_path = r'models\best_autoencoder_model.keras'
    forecaster_path = r'models\forecaster_model.keras'
    scaler_path = r'models\scaler.pkl'
    config_path = r'models\threshold.json'

    # 모델 로드
    yolo_model = YOLO(yolo_model_path)
    ae_model = tf.keras.models.load_model(ae_model_path)
    forecaster_model = tf.keras.models.load_model(forecaster_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        threshold = config['threshold']
    
    print(f" -> 로드 완료 (이상탐지 임계값: {threshold:.4f})")

    # 이미지 시퀀스 경로 수집 및 정렬
    search_path = r"dataset\**\*.jpg"
    image_paths = sorted(glob.glob(search_path, recursive=True))
    if not image_paths:
        print("에러: dataset 폴더에서 이미지를 찾을 수 없습니다.")
        return
        
    print(f"총 {len(image_paths)}장의 프레임을 로드했습니다. 무한 반복 재생을 시작합니다.")
    print("GUI 창이 열리면, 종료하시려면 창을 선택하고 'q' 키를 누르세요.")

    window_name = 'AI_1team Real-time Anomaly Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # 무한 반복 루프 (프레임 인덱스)
    idx = 0
    while True:
        img_path = image_paths[idx]
        img = cv2.imread(img_path)
        
        if img is None:
            idx = (idx + 1) % len(image_paths)
            continue
            
        img_area = img.shape[0] * img.shape[1]
        
        # [동적 튜닝] 낮과 밤을 시간대별로 구분하여 AI 확신도(Confidence)를 자동으로 조절합니다.
        current_hour = time.localtime().tm_hour
        if 19 <= current_hour or current_hour < 6:
            yolo_conf = 0.15
        else:
            yolo_conf = 0.25
            
        results = yolo_model(img, classes=[2, 5, 7], conf=yolo_conf, verbose=False)
        boxes = results[0].boxes
        
        car_count, bus_count, truck_count = 0, 0, 0
        total_bbox_area = 0.0
        
        # 바운딩 박스 그리기
        img_drawn = results[0].plot()
        
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 2: car_count += 1
            elif cls_id == 5: bus_count += 1
            elif cls_id == 7: truck_count += 1
            else: continue
                
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            total_bbox_area += (w * h)
            
        density = total_bbox_area / img_area if img_area > 0 else 0
        
        # 이상 탐지 (현재 저장된 모델이 LSTM Autoencoder이므로 3차원 사용)
        features = np.array([[car_count, bus_count, truck_count, density]], dtype=float)
        features_scaled = scaler.transform(features)
        
        # Autoencoder 이상탐지 예측 및 MSE 계산
        features_scaled_lstm = features_scaled.reshape(1, 1, 4)
        reconstructed = ae_model.predict(features_scaled_lstm, verbose=0)
        mse = np.mean(np.square(features_scaled_lstm - reconstructed), axis=(1, 2))[0]
        
        # 기본 이상 탐지 조건 (MSE가 임계값 초과)
        is_anomaly = mse > threshold
        
        # [예외 처리] 학습 데이터는 낮 시간대(차량 20~50대) 기준이므로,
        # 현재처럼 야간이나 새벽에 차량이 너무 적은 상태(예: 5대 미만)는 
        # 사고(Anomaly)가 아니라 단순히 '한산한 도로'로 간주하여 경고를 무시합니다.
        if (car_count + bus_count + truck_count) < 5:
            is_anomaly = False
        
        # 미래 밀집도 예측 (Traffic Forecasting - LSTM 모델이므로 3차원(1, 1, 4) 사용)
        predicted_density_scaled = forecaster_model.predict(features_scaled_lstm, verbose=0)[0][0]
        # 예측된 스케일링 값을 실제 밀집도 비율(0.0~1.0)로 역변환하기 위한 더미 배열 생성
        dummy_array = np.zeros((1, 4))
        dummy_array[0, 3] = predicted_density_scaled
        predicted_density_real = scaler.inverse_transform(dummy_array)[0, 3]
        
        # 알파 블렌딩(반투명) 오버레이 생성
        overlay = img_drawn.copy()
        
        if is_anomaly:
            status_text = f"[!] ANOMALY DETECTED (MSE: {mse:.4f})"
            color = (0, 0, 255) # BGR: 빨간색
        else:
            status_text = f"NORMAL TRAFFIC (MSE: {mse:.4f})"
            color = (0, 255, 0) # BGR: 초록색
            
        # 상단 배경에 반투명한 검정색 바(Bar) 깔기
        cv2.rectangle(overlay, (0, 0), (img_drawn.shape[1], 100), (0, 0, 0), -1)
        alpha = 0.6
        img_drawn = cv2.addWeighted(overlay, alpha, img_drawn, 1 - alpha, 0)
        
        # 텍스트 오버레이 렌더링
        cv2.putText(img_drawn, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        forecast_text = f"FORECAST | Next Density: {predicted_density_real:.3f}"
        cv2.putText(img_drawn, forecast_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 결과 이미지 띄우기 (버그 수정: frame 대신 bbox가 그려진 img_drawn 띄우기)
        cv2.imshow(window_name, img_drawn)
        
        # 'q' 키를 누르면 종료, 그 외에는 30ms 대기 후 다음 프레임
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("시연을 종료합니다.")
            break
            
        # 다음 프레임 (마지막 프레임이면 처음으로)
        idx = (idx + 1) % len(image_paths)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
