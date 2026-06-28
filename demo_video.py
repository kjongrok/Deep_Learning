import os
import cv2
import json
import joblib
import glob
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

def main():
    print("1. 모델 및 설정 로드 중...")
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

    # 이미지 시퀀스 경로 수집 및 정렬
    search_path = r"c:\AI_1team\dataset\**\*.jpg"
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
        
        # YOLO 객체 탐지
        results = yolo_model(img, verbose=False)
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
        
        # 이상 탐지 (Autoencoder)
        features = np.array([[car_count, bus_count, truck_count, density]], dtype=float)
        features_scaled = scaler.transform(features)
        reconstructed = ae_model.predict(features_scaled, verbose=0)
        mse = np.mean(np.square(features_scaled - reconstructed), axis=1)[0]
        
        is_anomaly = mse > threshold
        
        # 텍스트 오버레이 색상 및 문구
        if is_anomaly:
            status_text = f"STATUS: ANOMALY (MSE: {mse:.4f})"
            color = (0, 0, 255) # BGR: 빨간색
        else:
            status_text = f"STATUS: NORMAL (MSE: {mse:.4f})"
            color = (0, 255, 0) # BGR: 초록색
            
        # 검은색 배경 텍스트 박스 추가 (가독성 향상)
        cv2.rectangle(img_drawn, (10, 10), (700, 60), (0, 0, 0), -1)
        cv2.putText(img_drawn, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        # 화면 출력
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
