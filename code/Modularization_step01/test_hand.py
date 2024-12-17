import cv2
import mediapipe as mp
import numpy as np

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
        self.cap = cv2.VideoCapture(0)

        # 설정값
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        self.CATCH_THRESHOLD = 0.05
        self.MIN_HAND_LENGTH = 0.3
        self.prev_catch_state = [False, False]

    def calculate_distance(self, p1, p2):
        """두 랜드마크 사이의 거리를 계산합니다."""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def detect_catch(self, hand_landmarks):
        """
        엄지와 검지 끝 거리를 계산하여 잡기 동작을 감지합니다.

        Args:
            hand_landmarks: 손 랜드마크 객체

        Returns:
            True if 잡기 동작 감지, False otherwise
        """
        if hand_landmarks is None:
            return False

        try:
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
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

    def run(self):
        """핸드 트래킹을 실행합니다."""
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("웹캠을 찾을 수 없습니다.")
                continue

            results = self.process_frame(image)
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    middle_finger_tip = hand_landmarks.landmark[
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    ]
                    hand_length = self.calculate_distance(wrist, middle_finger_tip)

                    if hand_length < self.MIN_HAND_LENGTH:
                        continue

                    self.draw_hand_landmarks(image, hand_landmarks)
                    current_catch_state = self.detect_catch(hand_landmarks)

                    if current_catch_state:
                        cv2.putText(
                            image,
                            "CATCH",
                            (10 + hand_index * 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA,
                        )

                    if current_catch_state and not self.prev_catch_state[hand_index]:
                        print(f"CATCH (Hand {hand_index + 1})")

                    self.prev_catch_state[hand_index] = current_catch_state

                for hand_index in range(len(results.multi_hand_landmarks), 2):
                    self.prev_catch_state[hand_index] = False

            cv2.imshow("MediaPipe Hands", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_detection = HandDetection()
    hand_detection.run()
