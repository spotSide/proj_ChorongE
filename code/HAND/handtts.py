import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

#pip install pyttsx3

# TTS 엔진 초기화
engine = pyttsx3.init()

# 음성 속도 설정 (선택 사항)
engine.setProperty("rate", 175)  # 기본값은 200

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 자동 화이트 밸런스 설정 (선택 사항)
cap.set(cv2.CAP_PROP_AUTO_WB, 0.8)

# 손가락 잡기 감지 임계값 (엄지와 중지 끝 사이 거리)
CATCH_THRESHOLD = 0.15
# 이전 잡기 감지 상태 기록 변수 (터미널 출력용)
prev_catch_state = [False, False]
# 손 크기(손목-중지 길이) 제한 (정규화된 좌표 기준)
MIN_HAND_LENGTH = 0.3

def detect_catch(hand_landmarks, image_shape):
    """
    엄지와 중지 끝 거리를 계산하여 잡기 동작을 감지합니다.

    Args:
        hand_landmarks: 손 랜드마크 객체
        image_shape: 이미지의 (높이, 너비) 튜플

    Returns:
        True if 잡기 동작 감지, False otherwise
    """
    if hand_landmarks is None:
        return False

    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        distance = np.sqrt(
            (thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2
        )

        return distance < CATCH_THRESHOLD
    except:
        return False

def calculate_distance(p1, p2):
    """두 랜드마크 사이의 거리를 계산합니다."""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        continue

    # 색상 보정
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)
    cr = cv2.add(cr, 13)
    cb = cv2.subtract(cb, 20)
    image = cv2.merge([y, cr, cb])
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

    # 이미지에서 손 감지
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 손 크기 계산 및 필터링
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            hand_length = calculate_distance(wrist, middle_finger_tip)

            if hand_length < MIN_HAND_LENGTH:
                continue

            # 1. 손 그리기
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            # 2. 잡기 감지 (엄지와 중지 사용)
            current_catch_state = detect_catch(hand_landmarks, image.shape)

            # 화면에 CATCH 표시 및 TTS 출력
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

                # TTS로 "CATCH" 읽어주기
                if not prev_catch_state[hand_index]:  # 이전 상태가 False일 때만 읽어주기
                    engine.say("CATCH")
                    engine.runAndWait()
                    print(f"TTS: CATCH (Hand {hand_index + 1})")  # 터미널에 TTS 출력 메시지

            # 터미널에 CATCH 출력
            if current_catch_state and not prev_catch_state[hand_index]:
                print(f"CATCH (Hand {hand_index + 1})")

            prev_catch_state[hand_index] = current_catch_state

        # prev_catch_state 초기화
        for hand_index in range(len(results.multi_hand_landmarks), 2):
            prev_catch_state[hand_index] = False

    # 결과 이미지 보여주기
    cv2.imshow("MediaPipe Hands", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()