import os
import cv2
import mediapipe as mp
from ultralytics import YOLO

# YOLOv8 모델 로드 # YOLO("path/to/your/yolov8_model.pt")
yolo_model = YOLO('C:\\Pycharm_Project1\\Falldown_project\\Project_humanFall-main\\runs\\detect\\human_fall_s30\\weights\\best.pt')

data_root = 'C:\\Pycharm_Project1\\Falldown_project\\Data'
train_root = f'{data_root}\\Training\\SourceData\\Image'
image_paths = []

# MediaPipe Pose 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

for root, dirs, files in os.walk(train_root):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            if os.path.isfile(image_path):  # 파일 여부 확인
                image_paths.append(image_path)

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: 이미지 파일을 열 수 없습니다: {image_path}")
        continue

    # YOLOv8로 사람 감지
    results = yolo_model(image)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 감지된 사람 영역 추출
            person_image = image[y1:y2, x1:x2]

            # MediaPipe로 포즈 추정
            rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_image)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                desired_keypoints = [
                    landmarks[mp_pose.PoseLandmark.NOSE],  # 머리
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],  # 왼쪽 어깨
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],  # 오른쪽 어깨
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],  # 왼쪽 팔꿈치
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],  # 오른쪽 팔꿈치
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST],  # 왼쪽 손목
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],  # 오른쪽 손목
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE],  # 왼쪽 무릎
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],  # 오른쪽 무릎
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],  # 왼쪽 발목
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]  # 오른쪽 발목
                ]

                # 키포인트 그리기
                for keypoint in desired_keypoints:
                    cx, cy = int(keypoint.x * person_image.shape[1]), int(keypoint.y * person_image.shape[0])
                    cv2.circle(person_image, (cx, cy), 5, (0, 255, 0), -1)

                # 원본 이미지에 결과 적용
                image[y1:y2, x1:x2] = person_image

    # 결과 표시
    cv2.imshow('YOLOv8 + MediaPipe Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()