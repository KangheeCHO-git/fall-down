import os
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance
from tqdm import tqdm

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 랜드마크 인덱스 정의 # 11개
LANDMARKS = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28]

# 데이터 증강 함수 정의
def augment_sequence(sequence, factor=0.2):
    time_warped = []
    for landmark in sequence:
        x = np.arange(len(landmark))
        f = interp1d(x, landmark, kind='linear', axis=0)
        x_new = np.linspace(0, len(landmark) - 1, num=int(len(landmark) * (1 + factor)))
        time_warped.append(f(x_new))
    return np.array(time_warped)

# head_upper_body_speed 계산 함수
def calculate_head_upper_body_speed(keypoints, prev_keypoints):
    h = np.array([keypoints['landmark_0']['x'], keypoints['landmark_0']['y']])  # 머리 좌표
    l = np.array([keypoints['landmark_11']['x'], keypoints['landmark_11']['y']])  # 왼쪽 어깨 좌표
    r = np.array([keypoints['landmark_12']['x'], keypoints['landmark_12']['y']])  # 오른쪽 어깨 좌표

    # 이전 프레임의 좌표
    prev_h = np.array([prev_keypoints['landmark_0']['x'], prev_keypoints['landmark_0']['y']])
    prev_l = np.array([prev_keypoints['landmark_11']['x'], prev_keypoints['landmark_11']['y']])
    prev_r = np.array([prev_keypoints['landmark_12']['x'], prev_keypoints['landmark_12']['y']])

    # 현재 프레임과 이전 프레임의 상체 중심
    center_new = (h + l + r) / 3
    center_prev = (prev_h + prev_l + prev_r) / 3

    # 유클리드 거리 계산 (속도)
    dist_new = distance.euclidean(center_new, center_prev)
    return dist_new

# 데이터셋 클래스 정의
class FallSequenceDataset(Dataset):
    def __init__(self, json_files, sequence_length=3):
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        self.scaler = StandardScaler()

        all_landmarks = []

        for json_file in tqdm(json_files, desc="Processing JSON files"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            frames = list(data['pose_data'].values())

            for i in range(0, len(frames) - self.sequence_length + 1):
                sequence = frames[i:i + self.sequence_length]
                landmarks = []

                for j, frame in enumerate(sequence):
                    frame_landmarks = []
                    for landmark in LANDMARKS:
                        frame_landmarks.extend([
                            frame[f'landmark_{landmark}']['x'],
                            frame[f'landmark_{landmark}']['y']
                        ])
                    # YOLO xy ratio 및 머리/상체 속도 추가
                    bbox = frame.get('bbox', None)
                    if bbox:
                        yolo_xy_ratio = (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1'])
                    else:
                        yolo_xy_ratio = 0.0
                    frame_landmarks.append(yolo_xy_ratio)

                    if j > 0:
                        head_torso_speed = calculate_head_upper_body_speed(sequence[j], sequence[j - 1])
                    else:
                        head_torso_speed = 0.0
                    frame_landmarks.append(head_torso_speed)

                    landmarks.append(frame_landmarks)

                # 데이터 증강 적용
                augmented_sequence = augment_sequence(landmarks)
                all_landmarks.extend(augmented_sequence)

                # 레이블 재정의
                if sequence[-1]['class'] == 'Normal':
                    label = 0  # 비낙상
                elif sequence[-1]['class'] == 'Danger':
                    label = 1  # 낙상 위험
                elif sequence[-1]['class'] == 'Fall':
                    label = 2  # 완전 낙상

                self.sequences.append(augmented_sequence)
                self.labels.append(label)

        # 전체 데이터 정규화
        all_landmarks = np.array(all_landmarks)
        all_landmarks_scaled = self.scaler.fit_transform(all_landmarks)

        # 정규화된 데이터를 다시 시퀀스로 재구성
        for i in range(len(self.sequences)):
            start = i * self.sequence_length
            end = start + self.sequence_length
            self.sequences[i] = all_landmarks_scaled[start:end]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]]).squeeze()

# GRU 모델 정의
class FallDetectionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.5):
        super(FallDetectionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# 데이터 로드 및 전처리
json_folder = r'D:\Falldown\Re_video\re_landmark\re_train_NY'
json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
dataset = FallSequenceDataset(json_files)

# 데이터 로더 생성 전에 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(dataset.labels), y=dataset.labels)
class_weights = torch.FloatTensor(class_weights).to(device)

# 손실 함수에 가중치 적용
criterion = nn.CrossEntropyLoss(weight=class_weights)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

if len(dataset) > 0:
    sample_sequence, sample_label = dataset[0]
    input_size = sample_sequence.shape[1]

    model = FallDetectionGRU(input_size).to(device)
else:
    print("데이터 없음")
    exit()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
num_epochs = 500
best_loss = float('inf')
patience = 50  # 15 이후 다시 실행을 위해 50으로 변경
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    total_loss_train = 0

    for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss_train = criterion(outputs, labels.view(-1))
        loss_train.backward()
        optimizer.step()

        total_loss_train += loss_train.item()

    avg_loss_train = total_loss_train / len(train_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss_train:.4f}')

    scheduler.step(avg_loss_train)

    if avg_loss_train < best_loss:
        best_loss = avg_loss_train
        no_improve = 0
        torch.save(model.state_dict(), 'best_fall_detection_gru_3_p50.pt')
    else:
        no_improve += 1

    if no_improve >= patience:
        print("Early stopping")
        break

def calculate_metrics(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return f1, cm

# 학습 루프 내에서 성능 지표 계산
train_f1, train_cm = calculate_metrics(model, train_loader)
print(f'Train F1: {train_f1:.4f}')
print(f'Train CM:\n{train_cm}')

print("Training completed")
