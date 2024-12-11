import cv2
import mediapipe as mp
import numpy as np
import time

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
cap.set(cv2.CAP_PROP_AUTO_WB, 1)

# 손가락 잡기 감지 임계값 (엄지와 검지 끝 사이 거리)
CATCH_THRESHOLD = 0.05
# 이전 잡기 감지 상태 기록 변수 (터미널 출력용)
prev_catch_state = False

def detect_catch(hand_landmarks, image_shape):
    """
    엄지와 검지 끝 거리를 계산하여 잡기 동작을 감지합니다.

    Args:
        hand_landmarks: 손 랜드마크 객체
        image_shape: 이미지의 (높이, 너비) 튜플

    Returns:
        True if 잡기 동작 감지, False otherwise
    """
    if hand_landmarks is None:
        return False

    # 엄지 끝 & 검지 끝 좌표 가져오기
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] # 검지 끝으로 사용

        # 이미지 크기를 고려하여 거리 계산 (정규화된 좌표 사용)
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )

        # 잡기 감지 여부 반환
        return distance < CATCH_THRESHOLD
    except:
        return False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        continue

    # 색상 보정 (YCrCb 공간에서 조정)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)
    cr = cv2.add(cr, 13)
    cb = cv2.subtract(cb, 20)
    image = cv2.merge([y, cr, cb])
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

    # 이미지에서 손 감지 (RGB 변환 필요)
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image.flags.writeable = True

    # 이미지에 감지된 손 그리기 및 좌표 표시
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. 손 그리기
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            # 2. 각 랜드마크 좌표 표시
            image_height, image_width, _ = image.shape
            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * image_width), int(landmark.y * image_height)
                cv2.putText(
                    image,
                    f"({cx}, {cy})",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 3. 잡기 감지
            current_catch_state = detect_catch(hand_landmarks, image.shape)

            # 화면에 CATCH 표시 (잡고 있는 동안 계속 표시)
            if current_catch_state:
                cv2.putText(
                    image,
                    "CATCH",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            # 터미널에 CATCH 출력 (한 번만)
            if current_catch_state and not prev_catch_state:
                print("CATCH")

            prev_catch_state = current_catch_state

    # 결과 이미지 보여주기
    cv2.imshow("MediaPipe Hands", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()