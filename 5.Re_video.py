import os
import shutil
import glob
from tqdm import tqdm

# 경로 설정
train_json_dir = 'D:\\Falldown\\Re_video\\re_landmark\\re_train_NY_json'
val_json_dir = 'D:\\Falldown\\Re_video\\re_landmark\\re_val_NY_json'
train_target_video_dir = 'D:\\Falldown\\Re_video\\video\\re_train_NY'
val_target_video_dir = 'D:\\Falldown\\Re_video\\video\\re_val_NY'

# .mp4 비디오 파일 경로 패턴
video_pattern = r'D:\\Falldown\\Origin\\*\\SourceData\\Video\\*\\*\\*\\*.mp4'

# 대상 디렉토리 확인 및 생성
for target_dir in [train_target_video_dir, val_target_video_dir]:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# JSON 파일 이름 수집
train_json_files = glob.glob(os.path.join(train_json_dir, '*.json'))
val_json_files = glob.glob(os.path.join(val_json_dir, '*.json'))
train_json_filenames = {os.path.splitext(os.path.basename(f))[0] for f in train_json_files}
val_json_filenames = {os.path.splitext(os.path.basename(f))[0] for f in val_json_files}

# 일치하는 .mp4 비디오 파일 검색 및 복사
moved_files_count_train = 0
moved_files_count_val = 0

video_paths = glob.glob(video_pattern, recursive=True)
for video_path in tqdm(video_paths, desc="파일 이동 진행", unit="파일"):
    parent_folder_name = os.path.basename(os.path.dirname(video_path))
    
    if parent_folder_name in train_json_filenames:
        dest_path = os.path.join(train_target_video_dir, os.path.basename(video_path))
        shutil.copy(video_path, dest_path)
        moved_files_count_train += 1
    elif parent_folder_name in val_json_filenames:
        dest_path = os.path.join(val_target_video_dir, os.path.basename(video_path))
        shutil.copy(video_path, dest_path)
        moved_files_count_val += 1

# 이동된 파일 개수 출력
print(f"\n총 {moved_files_count_train}개의 파일이 {train_target_video_dir}로 이동되었습니다.")
print(f"총 {moved_files_count_val}개의 파일이 {val_target_video_dir}로 이동되었습니다.")
