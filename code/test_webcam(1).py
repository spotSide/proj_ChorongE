import cv2
from ultralytics import YOLO
import os

# 모델 로드
model = YOLO('/home/intel/문서/best.pt')

# 웹캠 열기 (기본 카메라: 0, 뎁스 카메라: 4)
cap = cv2.VideoCapture(0)

# 결과 이미지를 저장할 디렉토리를 생성 절대 경로로 ""안 수정해서 사용
output_dir = "detections"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 웹캠에서 프레임을 읽지 못한 경우 루프 종료
    
    # 모델에 프레임 전달
    results = model(frame)
    
    # 원본 프레임에 네모 박스 그리기
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # (x1, y1)부터 (x2, y2)까지 네모 박스 좌표
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 네모 박스 그리기
    
    # 결과 표시
    cv2.imshow("Detection", frame)
    
    # 프레임 저장
    frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1
    
    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 모든 윈도우 종료
cap.release()
cv2.destroyAllWindows()