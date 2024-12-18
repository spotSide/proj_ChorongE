import cv2
import mediapipe as mp
import numpy as np
import asyncio
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

            # 조건 확인
            if distance < self.PINKY_THRESHOLD:
                # 터미널에 CATCH 출력
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] CATCH - Pinky TIP near MCP!")
                return True

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

    async def run_hand_detection(self, shared_data):
        """비동기적으로 Hand Detection 실행"""
        print("Starting Hand Detection...")
        while shared_data['running']:
            frame = shared_data.get('frame')
            if frame is None:
                await asyncio.sleep(0)  # 이벤트 루프 양보
                continue

            # Hand Detection 처리
            image = frame.copy()
            results = self.process_frame(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.draw_hand_landmarks(image, hand_landmarks)
                    if self.detect_catch(hand_landmarks):
                        # 화면에 CATCH 텍스트 표시
                        cv2.putText(
                            image, "CATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2, cv2.LINE_AA
                        )

            # 화면 출력
            cv2.imshow("Hand Detection", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 종료 키 감지
                shared_data['running'] = False
                break
            await asyncio.sleep(0)  # 이벤트 루프 양보

        cv2.destroyAllWindows()


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
                if hand_detection.detect_catch(hand_landmarks):
                    # 화면에 CATCH 텍스트 표시
                    cv2.putText(
                        image, "CATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA
                    )

        cv2.imshow("Hand Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 종료 키 감지
            shared_data['running'] = False
            break
        await asyncio.sleep(0)  # 이벤트 루프 양보

    cv2.destroyAllWindows()
