import cv2
import mediapipe as mp
import numpy as np
import asyncio
import time
from datetime import datetime  # 현재 시간 출력을 위한 모듈 추가


class HandDetection:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 설정값
        self.PINKY_THRESHOLD = 0.05  # 새끼손가락 TIP과 MCP 사이 거리 임계값
        self.last_terminal_time = 0  # 마지막 터미널 출력 시간 기록

    def calculate_distance(self, p1, p2):
        """두 랜드마크 사이의 거리를 계산합니다."""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def detect_catch(self, hand_landmarks):
        """
        새끼손가락 TIP이 MCP (세 번째 마디)에 가까워졌을 때 catch 상태로 판별합니다.
        """
        if hand_landmarks is None:
            return False

        try:
            # 새끼손가락의 TIP과 MCP 좌표 가져오기
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

            # TIP과 MCP 사이 거리 계산
            distance = self.calculate_distance(pinky_tip, pinky_mcp)
            return distance < self.PINKY_THRESHOLD  # 임계값 이하이면 CATCH
        except Exception as e:
            print(f"Error in detect_catch: {e}")
            return False

    def process_frame(self, image):
        """프레임을 처리하고 손 랜드마크 및 동작을 감지합니다."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    def draw_hand_landmarks(self, image, hand_landmarks):
        """손 랜드마크를 이미지에 그립니다."""
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

    def handle_catch_display(self, image, hand_landmarks):
        """CATCH 상태를 처리: 오버레이와 터미널에 출력."""
        if self.detect_catch(hand_landmarks):
            # 오버레이: CATCH 텍스트 즉시 표시
            cv2.putText(
                image, "CATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA
            )

            # 터미널 출력: 1초에 한 번만 표시
            current_time = time.time()
            if current_time - self.last_terminal_time >= 1:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] CATCH - Pinky TIP near MCP!")
                self.last_terminal_time = current_time


async def run_hand_detection(shared_data):
    """비동기적으로 Hand Detection 실행"""
    hand_detection = HandDetection()

    while shared_data['running']:
        frame = shared_data.get('frame')
        if frame is None:
            await asyncio.sleep(0)  # 이벤트 루프 양보
            continue

        # Hand Detection 처리
        image = frame.copy()
        results = hand_detection.process_frame(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_detection.draw_hand_landmarks(image, hand_landmarks)
                hand_detection.handle_catch_display(image, hand_landmarks)

        cv2.imshow("Hand Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 종료 키 감지
            shared_data['running'] = False
            break
        await asyncio.sleep(0)  # 이벤트 루프 양보

    cv2.destroyAllWindows()
