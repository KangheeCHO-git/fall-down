#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from ultralytics import YOLO
import time
import yaml


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


# 데이터셋 경로 설정
data_root = 'D:\\Falldown\\Dataset\\Resized_Dataset'
train_root = f'{data_root}\\Train\\images'
val_root = f'{data_root}\\Val\\images'
test_root = f'{data_root}\\Test\\images'

# 클래스 설정
class_names = {0 : 'Non_Fall', 1 : 'Fall'}
num_classes = len(class_names)

# yaml 설정
yaml_info = {
    'path' : 'D:\\Falldown', # YOLOv8 기본 경로
    'names': class_names,
    'nc': num_classes,
    'train': train_root,
    'val': val_root,
    'test': test_root
}

# YAML 파일 저장 경로
yaml_file_path = 'D:\\Falldown\\yaml_info_yolov8s.yaml'

# YAML 파일 생성
with open(yaml_file_path, 'w') as f:
    yaml.dump(yaml_info, f)

print(f'yaml 파일이 생성되었습니다: {yaml_file_path}')


# In[ ]:


model = YOLO('yolov8s.pt')


# In[ ]:


# 학습 시작 시간 기록
start_time = time.time()

# 학습 실행
result = model.train(
    data='D:\\Falldown\\yaml_info_yolov8s.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device="cuda",
    workers=20,
    amp=True,
    patience=30,
    name='human_fall_s'
)

# 학습 종료 시간 기록
end_time = time.time()
execution_time = end_time - start_time
print(f"실행 시간: {execution_time:.4f} 초") # 약 20시간 소요


# In[ ]:


# 검증 시작 시간 기록
start_time = time.time()

# YOLO 모델 로드 및 검증 실행
model = YOLO('D:\\Falldown\\runs\\detect\\human_fall_s30\\weights\\best.pt')
val_results = model.val(
    data='D:\\Falldown\\yaml_info_yolov8s.yaml',
    imgsz=640,
    batch=32,
    device=device
)

# 평가 종료 시간 기록
end_time = time.time()
execution_time = end_time - start_time

# 결과 출력
print(f"Validation Results:")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"실행 시간: {execution_time:.4f} 초")


# In[ ]:


# 테스트 시작 시간 기록
start_time = time.time()

# 테스트 실행
test_results = model.val(
    data='D:\\Falldown\\yaml_info_yolov8s.yaml',
    imgsz=640,
    batch=32,
    device=device,
    split="test"  # Test 데이터셋으로 평가
)

# 테스트 종료 시간 기록
end_time = time.time()
execution_time = end_time - start_time

# 테스트 결과 출력
print(f"Test Results:")
print(f"mAP50: {test_results.box.map50:.4f}")
print(f"mAP50-95: {test_results.box.map:.4f}")
print(f"실행 시간: {execution_time:.4f} 초")

