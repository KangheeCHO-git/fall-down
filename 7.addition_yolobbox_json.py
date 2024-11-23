import os
import shutil
import glob
from tqdm import tqdm

# 디렉토리 경로 설정
train_video_dir = 'D:\\Falldown\\Re_video\\video\\re_train_NY'
val_video_dir = 'D:\\Falldown\\Re_video\\video\\re_val_NY'
json_search_dir = 'D:\\Falldown\\addition_yolobbox_json\\addition_yolobbox_json'
train_target_json_dir = 'D:\\Falldown\\Re_video\\addition_yolobbox_json\\re_train_NY'
val_target_json_dir = 'D:\\Falldown\\Re_video\\addition_yolobbox_json\\re_val_NY'

# 대상 디렉토리 확인 및 생성
for target_dir in [train_target_json_dir, val_target_json_dir]:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# .mp4 파일 이름 수집 (확장자 제외)
train_mp4_files = glob.glob(os.path.join(train_video_dir, '*.mp4'))
val_mp4_files = glob.glob(os.path.join(val_video_dir, '*.mp4'))
train_mp4_filenames = {os.path.splitext(os.path.basename(f))[0] for f in train_mp4_files}
val_mp4_filenames = {os.path.splitext(os.path.basename(f))[0] for f in val_mp4_files}

# JSON 파일 검색 및 이동
moved_files_count_train = 0
moved_files_count_val = 0

json_files = glob.glob(os.path.join(json_search_dir, '*.json'))
for json_path in tqdm(json_files, desc="JSON 파일 이동 진행", unit="파일"):
    json_filename = os.path.basename(json_path)
    file_basename, _ = os.path.splitext(json_filename)
    
    if file_basename in train_mp4_filenames:
        dest_path = os.path.join(train_target_json_dir, json_filename)
        shutil.copy(json_path, dest_path)
        moved_files_count_train += 1
    elif file_basename in val_mp4_filenames:
        dest_path = os.path.join(val_target_json_dir, json_filename)
        shutil.copy(json_path, dest_path)
        moved_files_count_val += 1

# 이동된 파일 개수 출력
print(f"\n총 {moved_files_count_train}개의 JSON 파일이 {train_target_json_dir}로 이동되었습니다.")
print(f"총 {moved_files_count_val}개의 JSON 파일이 {val_target_json_dir}로 이동되었습니다.")
