import cv2
import asyncio
from ultralytics import YOLO
import os
import logging
from datetime import datetime

# 로깅 수준 설정
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLODetector:
    def __init__(self, model_path='best_v4.pt'):
        # 모델 파일 경로 확인 및 로드
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

            # 원본 해상도 (1280x720)
            original_height, original_width = frame.shape[:2]

            # 중앙에서 320x320 크기로 자르기
            crop_x_start = (original_width - 320) // 2
            crop_y_start = (original_height - 480) // 2
            crop_x_end = crop_x_start + 320
            crop_y_end = crop_y_start + 480
            cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # 모델 예측
            results = self.model(cropped_frame, verbose=False)

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

                # 감지된 객체 바운딩 박스 표시 (cropped 영역 기준)
                cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    cropped_frame, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

            # 결과 표시
            output_frame = cv2.resize(cropped_frame, (640, 980))  # 출력창도 640x980으로 설정
            cv2.imshow("YOLO Detection", output_frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shared_data['running'] = False
                break

            await asyncio.sleep(0)  # 이벤트 루프 양보

        cv2.destroyAllWindows()
