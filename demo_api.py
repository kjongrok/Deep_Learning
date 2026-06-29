import os
import cv2
import json
import joblib
import time
import requests
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import winsound
import threading
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
ITS_API_KEY = os.getenv("ITS_API_KEY")
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase 클라우드 데이터베이스 연동 준비 완료!")
    except Exception as e:
        print(f"⚠️ Supabase 연결 실패: {e}")

def log_to_supabase_async(data):
    if not supabase_client: return
    def task():
        try:
            supabase_client.table("traffic_logs").insert(data).execute()
        except Exception as e:
            pass # 통신 실패 시 메인 프로그램에 영향을 주지 않음
    threading.Thread(target=task, daemon=True).start()

def get_cctv_url():
    # 카카오 지오코딩 로직 대신 기본 수원시 좌표로 진행
    minX, maxX, minY, maxY = '126.93', '127.09', '37.23', '37.33'
    loc_name = '수원시'
    road_type = 'ex'
    
    url = "https://openapi.its.go.kr:9443/cctvInfo"
    params = {
        'apiKey': ITS_API_KEY,
        'type': road_type,
        'cctvType': '1',  # 실시간 스트리밍
        'minX': minX,
        'maxX': maxX,
        'minY': minY,
        'maxY': maxY,
        'getType': 'json'
    }
    
    road_name_dict = {'ex': '고속도로', 'its': '일반국도', 'all': '전체 도로'}
    print(f"\n▶ ITS API 호출 중 ({loc_name} 인근 {road_name_dict[road_type]} CCTV 검색)...")
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cctv_list = data.get('response', {}).get('data', [])
            if cctv_list:
                if isinstance(cctv_list, dict):
                    cctv_list = [cctv_list]
                    
                print(f"\n✅ 총 {len(cctv_list)}개의 CCTV가 검색되었습니다.")
                for i, cctv in enumerate(cctv_list):
                    print(f"  {i+1}. {cctv['cctvname']}")
                    
                if len(cctv_list) == 1:
                    selected_idx = 0
                else:
                    while True:
                        choice = input(f"\n원하시는 CCTV 번호를 선택하세요 (1~{len(cctv_list)}): ").strip()
                        try:
                            selected_idx = int(choice) - 1
                            if 0 <= selected_idx < len(cctv_list):
                                break  # 올바른 번호를 입력하면 루프 탈출
                            else:
                                print(f"❌ 잘못된 번호입니다. 1에서 {len(cctv_list)} 사이의 숫자를 정확히 입력해주세요.")
                        except ValueError:
                            print("❌ 숫자가 아닙니다. 다시 입력해주세요.")
                        
                cctv = cctv_list[selected_idx]
                print(f"\n▶ [{cctv['cctvname']}] 영상에 연결합니다...")
                return cctv['cctvurl'], cctv['cctvname']
            else:
                print(f"❌ {loc_name} 인근에 해당하는 조건의 CCTV 영상을 찾을 수 없습니다.")
                return None, None
        else:
            print(f"❌ API 호출 실패 (상태 코드: {response.status_code})")
            return None, None
    except Exception as e:
        print(f"❌ API 요청 중 에러 발생: {e}")
        return None, None

def main():
    print("1. 모델 및 설정 로드 중...")
    yolo_model_path = r'notebooks\yolov8_nano.pt'
    ae_model_path = r'models/best_autoencoder_model.keras'
    forecaster_path = r'models/forecaster_model.keras'
    scaler_path = r'models/scaler.pkl'
    config_path = r'models/threshold.json'

    # 모델 로드
    yolo_model = YOLO(yolo_model_path)
    ae_model = tf.keras.models.load_model(ae_model_path)
    forecaster_model = tf.keras.models.load_model(forecaster_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        threshold = config['threshold']
    
    print(f" -> 로드 완료 (이상탐지 임계값: {threshold:.4f})\n")

    # CCTV URL 획득
    cctv_url, cctv_name = get_cctv_url()
    if not cctv_url:
        print("API 연동 실패로 인해 프로그램을 종료합니다.")
        return

    print(f"\n2. [{cctv_name}] 실시간 스트리밍 연결 중...")
    print("네트워크 상태에 따라 버퍼링이나 지연이 발생할 수 있습니다.")
    
    cap = cv2.VideoCapture(cctv_url)
    if not cap.isOpened():
        print("스트리밍 영상을 열 수 없습니다. URL 만료 또는 네트워크 이슈일 수 있습니다.")
        return

    window_name = 'AI_1team Real-time Anomaly Detection - Live CCTV'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("GUI 창이 열리며, 종료하시려면 창을 선택하고 'q' 키를 누르세요.")
    
    retry_count = 0
    max_retries = 30 

    while True:
        ret, img = cap.read()
        
        if not ret:
            print("⚠️ 프레임 수신 지연, 재시도 중...", end="\r")
            retry_count += 1
            if retry_count > max_retries:
                print("\n❌ 네트워크 연결이 끊어졌거나 스트리밍이 종료되었습니다.")
                break
            time.sleep(0.5)
            
            if retry_count % 10 == 0:
                print("\n▶ 스트림 재연결 시도 중...")
                cap.release()
                cap = cv2.VideoCapture(cctv_url)
            continue
            
        retry_count = 0 
            
        img_area = img.shape[0] * img.shape[1]
        if img_area == 0:
            continue
            
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
            
        density = total_bbox_area / img_area
        
        # 이상 탐지 (현재 저장된 모델이 LSTM Autoencoder이므로 3차원 사용)
        features = np.array([[car_count, bus_count, truck_count, density]], dtype=float)
        features_scaled = scaler.transform(features)
        
        features_scaled_lstm = features_scaled.reshape(1, 1, 4)
        reconstructed = ae_model.predict(features_scaled_lstm, verbose=0)
        mse = np.mean(np.square(features_scaled_lstm - reconstructed), axis=(1, 2))[0]
        
        is_anomaly = mse > threshold
        
        # [예외 처리] 학습 데이터는 낮 시간대(차량 20~50대) 기준이므로,
        # 현재처럼 야간이나 새벽에 차량이 너무 적은 상태(예: 5대 미만)는 
        # 사고(Anomaly)가 아니라 단순히 '한산한 도로'로 간주하여 경고를 무시합니다.
        if (car_count + bus_count + truck_count) < 5:
            is_anomaly = False
            
        # 미래 밀집도 예측 (Traffic Forecasting - LSTM 모델이므로 3차원(1, 1, 4) 사용)
        predicted_density_scaled = forecaster_model.predict(features_scaled_lstm, verbose=0)[0][0]
        dummy_array = np.zeros((1, 4))
        dummy_array[0, 3] = predicted_density_scaled
        predicted_density_real = scaler.inverse_transform(dummy_array)[0, 3]
            
        # Supabase에 1초(약 30프레임)마다 실시간 데이터 백그라운드 전송
        current_time = time.time()
        if not hasattr(log_to_supabase_async, "last_log_time"):
            log_to_supabase_async.last_log_time = 0
            
        if current_time - log_to_supabase_async.last_log_time >= 1.0:
            log_data = {
                "car_count": int(car_count),
                "bus_count": int(bus_count),
                "truck_count": int(truck_count),
                "density": float(density),
                "anomaly_mse": float(mse),
                "is_anomaly": bool(is_anomaly),
                "predicted_next_density": float(predicted_density_real)
            }
            log_to_supabase_async(log_data)
            log_to_supabase_async.last_log_time = current_time
        
        # 알파 블렌딩 오버레이
        overlay = img_drawn.copy()
        
        if is_anomaly:
            status_text = f"[!] ANOMALY DETECTED (MSE: {mse:.4f})"
            color = (0, 0, 255) # BGR: 빨간색
            winsound.Beep(1000, 100)
        else:
            status_text = f"NORMAL TRAFFIC (MSE: {mse:.4f})"
            color = (0, 255, 0) # BGR: 초록색
            
        cv2.rectangle(overlay, (0, 0), (img_drawn.shape[1], 100), (0, 0, 0), -1)
        alpha = 0.6
        img_drawn = cv2.addWeighted(overlay, alpha, img_drawn, 1 - alpha, 0)
        
        cv2.putText(img_drawn, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        # 미래 예측 상태 텍스트
        forecast_text = f"FORECAST | Next Density: {predicted_density_real:.3f}"
        cv2.putText(img_drawn, forecast_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 결과 이미지 띄우기
        cv2.imshow(window_name, img_drawn)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("시연을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
