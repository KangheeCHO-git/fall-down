import os
import shutil
import glob
from tqdm import tqdm

# JSON 파일 경로 설정
train_json_dir = 'D:\\Falldown\\Re_video\\re_landmark\\re_train_NY'
val_json_dir = 'D:\\Falldown\\Re_video\\re_landmark\\re_val_NY'
train_target_json_dir = 'D:\\Falldown\\Re_video\\json\\re_train_NY'
val_target_json_dir = 'D:\\Falldown\\Re_video\\json\\re_val_NY'

# 원본 JSON 파일 경로 패턴
json_pattern = r'D:\\Falldown\\Origin\\*\\LabelingData\\Video\\*\\*\\*\\*.json'

# 대상 디렉토리 확인 및 생성
for target_dir in [train_target_json_dir, val_target_json_dir]:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# JSON 파일 이름 수집
train_json_files = glob.glob(os.path.join(train_json_dir, '*.json'))
val_json_files = glob.glob(os.path.join(val_json_dir, '*.json'))
train_json_filenames = {os.path.splitext(os.path.basename(f))[0] for f in train_json_files}
val_json_filenames = {os.path.splitext(os.path.basename(f))[0] for f in val_json_files}

# 일치하는 JSON 파일 검색 및 복사
moved_files_count_train = 0
moved_files_count_val = 0

json_paths = glob.glob(json_pattern, recursive=True)
for json_path in tqdm(json_paths, desc="JSON 파일 이동 진행", unit="파일"):
    json_filename = os.path.basename(json_path)
    file_basename, _ = os.path.splitext(json_filename)
    
    if file_basename in train_json_filenames:
        dest_path = os.path.join(train_target_json_dir, json_filename)
        shutil.copy(json_path, dest_path)
        moved_files_count_train += 1
    elif file_basename in val_json_filenames:
        dest_path = os.path.join(val_target_json_dir, json_filename)
        shutil.copy(json_path, dest_path)
        moved_files_count_val += 1

# 이동된 파일 개수 출력
print(f"\n총 {moved_files_count_train}개의 JSON 파일이 {train_target_json_dir}로 이동되었습니다.")
print(f"총 {moved_files_count_val}개의 JSON 파일이 {val_target_json_dir}로 이동되었습니다.")
