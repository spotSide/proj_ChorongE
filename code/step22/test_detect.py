import cv2
import asyncio
from ultralytics import YOLO
import os
import logging
from datetime import datetime

# 로깅 수준 설정
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLODetector:
    def __init__(self, model_path='best.pt'):
        # 현재 파일이 위치한 디렉토리에서 best.pt 파일을 찾습니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = YOLO(model_path)
        self.last_detection_time = 0  # 마지막 출력 시간을 기록

    async def run_detection(self, shared_data):
        """비동기적으로 YOLO 모델을 사용해 객체 감지를 실행합니다."""
        print("Starting YOLO Detection...")
        while shared_data['running']:
            frame = shared_data.get('frame')
            if frame is None:
                await asyncio.sleep(0)  # 프레임이 준비될 때까지 대기
                continue

            # 이미지 전처리 (YOLO 입력 해상도: 640x640)
            img_resized = cv2.resize(frame, (640, 640))

            # 모델 예측
            results = self.model(img_resized, verbose=False)

            # 현재 시간
            current_time = asyncio.get_event_loop().time()

            # 감지된 객체 출력
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls)  # 클래스 ID 가져오기
                class_name = self.model.names[class_id]  # 클래스 이름 가져오기

                # 초당 1회만 출력
                if current_time - self.last_detection_time >= 1:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{now}] Detected: {class_name}")
                    self.last_detection_time = current_time

            # 감지 결과 시각화
            plots = results[0].plot()

            # 감지 결과를 1280x720 해상도로 조정
            output_frame = cv2.resize(plots, (1280, 720))

            # 결과 표시
            cv2.imshow("YOLO Detection", output_frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shared_data['running'] = False
                break

            await asyncio.sleep(0)  # 이벤트 루프 양보

        cv2.destroyAllWindows()

# print(f"Ultralytics YOLO version: {__version__}")
