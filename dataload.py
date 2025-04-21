import kagglehub
import shutil
from pathlib import Path

downloaded_path = kagglehub.dataset_download("mohammedtgadallah/mt-small-dataset")
print("Downloaded to:", downloaded_path)

target_path = Path(".")  # 원하는 경로로 수정
target_path.mkdir(parents=True, exist_ok=True)

# 3. 데이터 복사
shutil.copytree(downloaded_path, target_path, dirs_exist_ok=True)
print("Copied to:", target_path)
