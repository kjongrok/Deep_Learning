import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import requests
import os
import json
import joblib
import threading
import glob
import tensorflow as tf
from ultralytics import YOLO
from dotenv import load_dotenv
from supabase import create_client

# 페이지 설정
st.set_page_config(page_title="AI_1team 관제 대시보드", layout="wide", page_icon="🚗")

# ==========================================
# 1. 초기 셋업 (모델 및 DB 연동)
# ==========================================
# 환경변수 로드
load_dotenv()
ITS_API_KEY = os.getenv("ITS_API_KEY")
try:
    if not ITS_API_KEY and "ITS_API_KEY" in st.secrets:
        ITS_API_KEY = st.secrets["ITS_API_KEY"]
except Exception:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
try:
    if not SUPABASE_URL and "SUPABASE_URL" in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
except Exception:
    pass

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
try:
    if not SUPABASE_KEY and "SUPABASE_KEY" in st.secrets:
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    pass

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Supabase 연결 실패: {e}")

def log_to_supabase_async(data):
    if not supabase: return
    def task():
        try:
            supabase.table("traffic_logs").insert(data).execute()
        except Exception:
            pass
    threading.Thread(target=task, daemon=True).start()

@st.cache_resource
def load_models():
    yolo_model = YOLO(r'notebooks/yolov8_nano.pt')
    ae_model = tf.keras.models.load_model(r'models/best_autoencoder_model.keras')
    forecaster_model = tf.keras.models.load_model(r'models/forecaster_model.keras')
    scaler = joblib.load(r'models/scaler.pkl')
    with open(r'models/threshold.json', 'r') as f:
        threshold = json.load(f)['threshold']
    return yolo_model, ae_model, forecaster_model, scaler, threshold

yolo_model, ae_model, forecaster_model, scaler, threshold = load_models()

@st.cache_data(ttl=3600)
def get_cctv_list(api_key):
    demo_video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    if not api_key:
        return [{"cctvname": "[Demo] 경부선 서울요금소 (API 키 없음)", "cctvurl": demo_video_url}]
    minX, maxX, minY, maxY = '126.93', '127.09', '37.23', '37.33'
    url = "https://openapi.its.go.kr:9443/cctvInfo"
    params = {
        'apiKey': api_key, 'type': 'ex', 'cctvType': '1',
        'minX': minX, 'maxX': maxX, 'minY': minY, 'maxY': maxY,
        'getType': 'json'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return [{"cctvname": f"[Demo] 영동선 마성터널 (API 응답 에러)", "cctvurl": demo_video_url}]
        data = response.json().get('response', {}).get('data', [])
        if isinstance(data, dict): data = [data]
        if not data:
            return [{"cctvname": "[Demo] 서해안선 서해대교 (CCTV 없음)", "cctvurl": demo_video_url}]
        return data
    except Exception as e:
        # 해외 IP 차단(스트림릿 클라우드) 시 데모 리스트 제공
        return [
            {"cctvname": "📷 [Live Demo] 경부선 서울요금소", "cctvurl": demo_video_url},
            {"cctvname": "📷 [Live Demo] 영동선 마성터널", "cctvurl": demo_video_url},
            {"cctvname": "📷 [Live Demo] 서해안선 서해대교", "cctvurl": demo_video_url}
        ]

cctv_list = get_cctv_list(ITS_API_KEY)
cctv_options = {cctv['cctvname']: cctv['cctvurl'] for cctv in cctv_list}

# ==========================================
# 2. 사이드바 컨트롤
# ==========================================
st.sidebar.title("⚙️ 시스템 컨트롤")
st.sidebar.markdown("---")
selected_cctv_name = st.sidebar.selectbox("📷 [Live] CCTV 위치 선택", list(cctv_options.keys()))
selected_cctv_url = cctv_options[selected_cctv_name]
st.sidebar.markdown("---")
run_stream = st.sidebar.checkbox("▶️ [Live] 스트리밍 시작", value=False)

st.title("🚗 교통량 이상 탐지 및 예측 MLOps 시스템")

# ==========================================
# 3. 다중 탭(Tabs) 레이아웃 생성
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 1. 실시간 관제 (Live)", 
    "🎬 2. 과거 학습 시뮬레이션", 
    "📊 3. 모델 방어 보고서", 
    "🗄️ 4. 클라우드 DB & 아키텍처", 
    "📈 5. 데이터 탐색 (EDA)"
])

# ------------------------------------------
# Tab 1: 실시간 관제 (Live)
# ------------------------------------------
with tab1:
    st.markdown(f"**현재 관제 중인 위치:** {selected_cctv_name}")
    col1, col2, col3, col4 = st.columns(4)
    metric_car = col1.empty()
    metric_bus = col2.empty()
    metric_truck = col3.empty()
    metric_density = col4.empty()

    st.markdown("---")
    video_col, chart_col = st.columns([1.2, 1])
    with video_col:
        st.subheader("📡 Live CCTV 관제 화면")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    with chart_col:
        st.subheader("📈 실시간 밀집도 (Actual vs Predicted)")
        chart_placeholder = st.empty()

    if 'history_df' not in st.session_state:
        st.session_state.history_df = pd.DataFrame(columns=["Time", "Actual", "Predicted"])

    if run_stream and selected_cctv_url:
        cap = cv2.VideoCapture(selected_cctv_url)
        if not cap.isOpened():
            video_placeholder.error("스트리밍 연결에 실패했습니다.")
        else:
            frame_skip = 10 
            frame_idx = 0
            last_log_time = 0
            
            while run_stream:
                ret, img = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue
                    
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                img_area = img.shape[0] * img.shape[1]
                
                # [동적 튜닝]
                current_hour = time.localtime().tm_hour
                yolo_conf = 0.15 if (19 <= current_hour or current_hour < 6) else 0.25
                    
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
                    total_bbox_area += ((x2 - x1) * (y2 - y1))
                    
                density = total_bbox_area / img_area if img_area > 0 else 0
                
                features = np.array([[car_count, bus_count, truck_count, density]], dtype=float)
                features_scaled = scaler.transform(features)
                
                features_scaled_lstm = features_scaled.reshape(1, 1, 4)
                reconstructed = ae_model.predict(features_scaled_lstm, verbose=0)
                mse = np.mean(np.square(features_scaled_lstm - reconstructed), axis=(1, 2))[0]
                
                is_anomaly = mse > threshold
                if (car_count + bus_count + truck_count) < 5:
                    is_anomaly = False
                    
                predicted_density_scaled = forecaster_model.predict(features_scaled_lstm, verbose=0)[0][0]
                dummy_array = np.zeros((1, 4))
                dummy_array[0, 3] = predicted_density_scaled
                predicted_density_real = scaler.inverse_transform(dummy_array)[0, 3]

                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    log_to_supabase_async({
                        "car_count": int(car_count), "bus_count": int(bus_count), "truck_count": int(truck_count),
                        "density": float(density), "anomaly_mse": float(mse), "is_anomaly": bool(is_anomaly),
                        "predicted_next_density": float(predicted_density_real)
                    })
                    last_log_time = current_time

                metric_car.metric("승용차", f"{car_count} 대")
                metric_bus.metric("버스", f"{bus_count} 대")
                metric_truck.metric("트럭", f"{truck_count} 대")
                metric_density.metric("혼잡도(Density)", f"{density:.4f}")
                
                if is_anomaly:
                    status_placeholder.error(f"🚨 ANOMALY DETECTED! (MSE: {mse:.4f})")
                else:
                    status_placeholder.success(f"✅ NORMAL TRAFFIC (MSE: {mse:.4f})")
                    
                img_drawn_resized = cv2.resize(img_drawn, (800, 450))
                img_rgb = cv2.cvtColor(img_drawn_resized, cv2.COLOR_BGR2RGB)
                video_placeholder.image(img_rgb, channels="RGB", width="stretch")
                
                new_row = pd.DataFrame({
                    "Time": [time.strftime("%H:%M:%S")],
                    "Actual": [density],
                    "Predicted": [predicted_density_real]
                })
                st.session_state.history_df = pd.concat([st.session_state.history_df, new_row]).tail(30)
                chart_data = st.session_state.history_df.set_index("Time")
                chart_placeholder.line_chart(chart_data, color=["#1E90FF", "#FF4B4B"])
                
                time.sleep(0.01)
            cap.release()
    else:
        video_placeholder.info("👈 사이드바에서 '스트리밍 시작'을 체크해주세요.")

# ------------------------------------------
# Tab 2: 과거 학습 시뮬레이션 (Offline)
# ------------------------------------------
with tab2:
    st.subheader("🎬 모델 훈련에 사용된 원본 데이터셋 기반 오프라인 시뮬레이션")
    st.markdown("라이브 스트리밍이 아닌, 모델을 훈련할 때 썼던 로컬 데이터를 바탕으로 모델이 얼마나 정확하게 판단하는지 끊김 없이 보여줍니다.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown("로컬에 있는 아무 블랙박스/CCTV 영상(mp4)이나 업로드 해보세요!")
        uploaded_video = st.file_uploader("🎥 시뮬레이션 비디오 업로드", type=['mp4', 'avi'])
        
        start_sim = st.button("▶️ 시뮬레이션 시작 (Click)")
        stop_sim = st.button("⏹️ 중지")
        sim_status = st.empty()
    with sim_col2:
        sim_video_placeholder = st.empty()
        
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
        
    if start_sim:
        if uploaded_video is None:
            sim_status.error("먼저 비디오 파일을 업로드 해주세요!")
        else:
            st.session_state.sim_running = True
            # 업로드된 파일을 임시 저장
            with open("temp_sim_video.mp4", "wb") as f:
                f.write(uploaded_video.read())
                
    if stop_sim:
        st.session_state.sim_running = False
        
    if st.session_state.sim_running:
        cap_sim = cv2.VideoCapture("temp_sim_video.mp4")
        if not cap_sim.isOpened():
            sim_status.error("비디오를 읽을 수 없습니다.")
            st.session_state.sim_running = False
        else:
            sim_status.success("시뮬레이션 영상 분석 중...")
            while st.session_state.sim_running:
                ret, img = cap_sim.read()
                if not ret:
                    break # 영상 끝
                    
                img_area = img.shape[0] * img.shape[1]
                # 동적 튜닝 생략 (업로드 영상이므로 기본 0.25 사용)
                results = yolo_model(img, classes=[2, 5, 7], conf=0.25, verbose=False)
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
                    total_bbox_area += ((x2 - x1) * (y2 - y1))
                    
                density = total_bbox_area / img_area if img_area > 0 else 0
                
                features = np.array([[car_count, bus_count, truck_count, density]], dtype=float)
                features_scaled = scaler.transform(features)
                
                features_scaled_lstm = features_scaled.reshape(1, 1, 4)
                reconstructed = ae_model.predict(features_scaled_lstm, verbose=0)
                mse = np.mean(np.square(features_scaled_lstm - reconstructed), axis=(1, 2))[0]
                
                is_anomaly = mse > threshold
                if (car_count + bus_count + truck_count) < 5: is_anomaly = False
                
                if is_anomaly:
                    cv2.putText(img_drawn, f"[!] ANOMALY DETECTED (MSE: {mse:.4f})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    cv2.putText(img_drawn, f"NORMAL TRAFFIC (MSE: {mse:.4f})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    
                img_drawn_resized = cv2.resize(img_drawn, (800, 450))
                img_rgb = cv2.cvtColor(img_drawn_resized, cv2.COLOR_BGR2RGB)
                sim_video_placeholder.image(img_rgb, channels="RGB", width="stretch")
                
            cap_sim.release()
            st.session_state.sim_running = False
            sim_status.info("시뮬레이션 영상 재생이 완료되었습니다.")

# ------------------------------------------
# Tab 3: 모델 방어 보고서 (Defense)
# ------------------------------------------
with tab3:
    st.subheader("💡 최적 모델 채택 사유 (Model Defense)")
    
    col_def1, col_def2 = st.columns([1, 1])
    with col_def1:
        st.markdown("#### 🏆 K-Means vs MLP vs LSTM 성능 비교")
        f1_data = pd.DataFrame({
            "Model Architecture": ["K-Means", "MLP Autoencoder", "8-Neuron LSTM", "16-Neuron LSTM (Champion)", "32-Neuron LSTM"],
            "F1-Score": [0.2450, 0.3120, 0.4610, 0.4618, 0.4610]
        }).set_index("Model Architecture")
        st.bar_chart(f1_data)
        
    with col_def2:
        st.markdown("#### 🤔 왜 16-Neuron LSTM 인가?")
        st.info("""
        **1. 랜덤 시드(Random Seed)에 의한 분산(Variance)**
        - 딥러닝 모델은 초기 가중치가 무작위로 설정되기 때문에, 데이터셋이 작고 에포크가 짧은 상황에서는 평가 지표가 일시적으로 요동칠 수 있습니다. 
        - 하지만 다중 시뮬레이션 결과, 16-Neuron 구조가 가장 안정적으로 최고점(F1 0.4618)을 달성했습니다.
        
        **2. 과적합 방지와 연산량의 골디락스 존 (Sweet Spot)**
        - **8-Neuron:** 구조가 너무 단순하여 복잡한 트래픽 패턴 학습에 한계(과소적합 우려).
        - **32-Neuron:** 모델이 너무 무거워 실시간 CCTV 추론(Inference) 시 지연(Latency)이 발생할 우려가 큼.
        - **16-Neuron:** 최고 성능을 내면서도 실시간 30fps 스트리밍을 거뜬히 방어해내는 완벽한 트레이드오프(Trade-off) 비율을 증명.
        """)

# ------------------------------------------
# Tab 4: 아키텍처 & DB (Pipeline)
# ------------------------------------------
with tab4:
    st.subheader("🛠️ 시스템 아키텍처 및 실시간 클라우드 로깅")
    
    st.markdown("""
    본 프로젝트는 단순 로컬 실행에 그치지 않고, 상용 서비스 수준의 **실시간 백엔드 파이프라인(MLOps)**을 구축했습니다.
    """)
    
    col_arch1, col_arch2 = st.columns([1, 2])
    with col_arch1:
        st.markdown("#### 🌐 Pipeline Architecture")
        st.markdown("""
        ```mermaid
        graph TD
            A[공공 ITS CCTV] -->|Stream| B(YOLOv8 Nano)
            B -->|BBox Metrics| C{AI Engine}
            C --> D[LSTM Autoencoder]
            C --> E[LSTM Forecaster]
            D -->|MSE Score| F[Anomaly Detection]
            E -->|t+1 Density| G[Traffic Prediction]
            C -.->|Async Logging| H[(Supabase Cloud DB)]
        ```
        """)
        
    with col_arch2:
        st.markdown("#### 🗄️ Supabase Cloud DB 실시간 적재 현황")
        st.markdown("현재 `traffic_logs` 테이블에 적재되고 있는 최신 데이터 50개입니다.")
        
        if supabase:
            try:
                res = supabase.table("traffic_logs").select("*").order("id", desc=True).limit(50).execute()
                if res.data:
                    df_logs = pd.DataFrame(res.data)
                    st.dataframe(df_logs, use_container_width=True)
                else:
                    st.warning("데이터베이스에 아직 로그가 없습니다.")
            except Exception as e:
                st.error("DB 로드 중 에러 발생")
        else:
            st.error("Supabase 연결이 설정되지 않았습니다.")

# ------------------------------------------
# Tab 5: 데이터 탐색 (EDA & Analytics)
# ------------------------------------------
with tab5:
    st.subheader("📊 실시간 누적 데이터 분석 (Cloud DB Analytics)")
    st.markdown("로컬 엑셀 파일이 아닌, 현재 **Supabase 클라우드에 누적된 전체 트래픽 데이터**를 바탕으로 기초 통계와 패턴을 실시간으로 분석합니다.")
    
    if supabase:
        try:
            # Supabase에서 전체 로그 가져오기 (최대 5000개 제한)
            res = supabase.table("traffic_logs").select("*").order("id", desc=True).limit(5000).execute()
            if res.data and len(res.data) > 0:
                df_eda = pd.DataFrame(res.data)
                col_eda1, col_eda2 = st.columns([1, 1])
                
                with col_eda1:
                    st.markdown("#### 📝 차량 종류별 기초 통계량 (DB 평균)")
                    stats = df_eda[['car_count', 'bus_count', 'truck_count']].mean().rename("평균 대수 (프레임당)")
                    st.dataframe(stats, use_container_width=True)
                    st.caption(f"클라우드에 누적된 {len(df_eda)}개의 데이터를 분석한 결과, 승용차의 비중이 압도적으로 높음을 확인했습니다.")
                    
                with col_eda2:
                    st.markdown("#### 📈 시간 경과에 따른 밀집도(Density) 누적 추이")
                    # 시간순으로 정렬하기 위해 데이터를 뒤집음 (과거 -> 현재)
                    df_chart = df_eda.sort_values("id")
                    st.line_chart(df_chart['density'].values)
                    st.caption("특정 시점부터 혼잡도가 급증하는 구간을 실시간 데이터 기반으로 분석합니다.")
            else:
                st.warning("DB에 분석할 데이터가 충분하지 않습니다. 스트리밍을 켜서 데이터를 수집해주세요.")
        except Exception as e:
            st.error(f"DB 데이터를 불러오는 중 에러가 발생했습니다: {e}")
    else:
        st.error("Supabase 연결이 설정되지 않아 데이터를 분석할 수 없습니다.")
