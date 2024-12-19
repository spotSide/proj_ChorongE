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
        self.detection_flag = False  # 감지 상태 플래그
        self.flag_reset_time = 0  # 플래그 유지 종료 시간

    async def manage_detection_flag(self):
        """비동기로 감지 플래그를 관리합니다."""
        # print("Starting manage_detection_flag...")  # 디버깅 출력
        self.detection_flag = True
        print("class detect flag - 5s")  # 플래그 활성화 출력
        self.flag_reset_time = asyncio.get_event_loop().time() + 5  # 현재 시간 기준 5초 후 해제
        await asyncio.sleep(5)  # 5초 유지
        self.detection_flag = False
        print("class flag end")  # 플래그 종료 출력

    async def run_detection(self, shared_data):
        """비동기적으로 YOLO 모델을 사용해 객체 감지를 실행합니다."""
        print("Starting YOLO Detection...")
        while shared_data['running']:
            frame = shared_data.get('frame')
            if frame is None:
                await asyncio.sleep(0)  # 프레임이 준비될 때까지 대기
                continue

            # 중앙에서 320x480 크기로 자르기
            original_height, original_width = frame.shape[:2]
            crop_x_start = (original_width - 320) // 2
            crop_y_start = (original_height - 480) // 2
            crop_x_end = crop_x_start + 320
            crop_y_end = crop_y_start + 480
            cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # 모델 예측
            results = self.model(cropped_frame, verbose=False)

            # 현재 시간
            current_time = asyncio.get_event_loop().time()

            # YOLO의 바운딩 박스 및 확률 그대로 표시
            for box, cls, score in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls)  # 클래스 ID 가져오기
                class_name = self.model.names[class_id]  # 클래스 이름 가져오기

                # 초당 1회만 터미널 출력
                if current_time - self.last_detection_time >= 1:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{now}] Detected: {class_name} ({score:.2f})")
                    self.last_detection_time = current_time

                # 플래그 설정 (새로운 감지 시 비동기 관리 태스크 실행)
                if not self.detection_flag:
                    # print("Creating detection flag task...")  # 디버깅 출력
                    asyncio.create_task(self.manage_detection_flag())

                # YOLO 바운딩 박스 및 확률 표시
                cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    cropped_frame, f"{class_name} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

            # 감지 상태 유지 중이면 표시
            if self.detection_flag:
                cv2.putText(
                    cropped_frame, " ", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                )

            # 결과 표시
            output_frame = cv2.resize(cropped_frame, (640, 480))  # 출력창도 640x480으로 설정
            cv2.imshow("YOLO Detection", output_frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shared_data['running'] = False
                break

            await asyncio.sleep(0)  # 이벤트 루프 양보

        cv2.destroyAllWindows()
