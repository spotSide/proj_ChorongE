import cv2
import mediapipe as mp
import numpy as np
import asyncio

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
        self.CATCH_THRESHOLD = 0.05
        self.MIN_HAND_LENGTH = 0.3
        self.prev_catch_state = [False, False]

    def calculate_distance(self, p1, p2):
        """두 랜드마크 사이의 거리를 계산합니다."""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def detect_catch(self, hand_landmarks):
        """잡기 동작을 감지합니다."""
        if hand_landmarks is None:
            return False

        try:
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = self.calculate_distance(thumb_tip, index_tip)
            return distance < self.CATCH_THRESHOLD
        except:
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

    def run(self, shared_data=None):
        """핸드 트래킹 실행 (shared_data를 사용하거나 웹캠을 직접 사용)."""
        while True:
            try:
                if shared_data:
                    # shared_data에서 프레임 가져오기
                    frame = shared_data.get('frame')
                    if frame is None:
                        continue
                    image = frame.copy()
                else:
                    raise ValueError("Shared data is required for this run.")

            except ValueError as e:
                print(e)
                break

            results = self.process_frame(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.draw_hand_landmarks(image, hand_landmarks)
                    if self.detect_catch(hand_landmarks):
                        cv2.putText(
                            image,
                            "CATCH",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            cv2.imshow("Hand Detection", image)
            if cv2.waitKey(5) & 0xFF == ord('q'):  # `q` 키로 종료
                shared_data['running'] = False
                break

        cv2.destroyAllWindows()


# def run_hand_detection(shared_data):
#     hand_detection = HandDetection()
#     while shared_data['running']:
#         frame = shared_data.get('frame')
#         if frame is None:
#             continue  # 프레임이 없으면 건너뜀
        
#         image = frame.copy()
#         hand_detection_results = hand_detection.process_frame(image)

#         if hand_detection_results.multi_hand_landmarks:
#             for landmarks in hand_detection_results.multi_hand_landmarks:
#                 hand_detection.draw_hand_landmarks(image, landmarks)
#                 if hand_detection.detect_catch(landmarks):
#                     cv2.putText(image, "CATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1.0, (0, 0, 255), 2, cv2.LINE_AA)

#         cv2.imshow("Hand Detection", image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 종료 키 감지
#             shared_data['running'] = False
#             break

#     cv2.destroyAllWindows()

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
                    cv2.putText(
                        image, "CATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA
                    )

        cv2.imshow("Hand Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared_data['running'] = False
            break
        await asyncio.sleep(0)  # 이벤트 루프 양보



