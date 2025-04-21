from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import torchvision.transforms as T
import numpy as np
from utils import img_load, split_fixed_test, save_fold_split_index, log_fold_score, show_cv_summary_if_complete
from utils import evaluate_and_save_metrics, visualize_and_save_predictions, save_loss_curve, dice_score
from SegDataset import SegDataset
from model import AttentionUNet

from sklearn.model_selection import KFold
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="K-Fold Training")
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0 ~ K-1)')
    args = parser.parse_args()
    return args
# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# 몇 개의 GPU가 있는지
print("Available GPUs:", torch.cuda.device_count())

# 현재 활성화된 GPU 이름
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**2, 1), "MB")
    print("Memory Cached:   ", round(torch.cuda.memory_reserved(0)/1024**2, 1), "MB")

K = 5
img_dir = Path("./MT-Small-Dataset/Benign/Original_Benign")
mask_dir = Path("./MT-Small-Dataset/Benign/Ground_Truth_Benign")

## 세팅!
save_root = './results/unet'
num_epochs = 50
args = get_args()
fold_idx = args.fold
seed = 42


paired_files = img_load(img_dir=img_dir, mask_dir=mask_dir)

train_val_pairs, test_pairs = split_fixed_test(paired_files, seed=seed)
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
indices = np.arange(len(train_val_pairs))
folds = list(kf.split(indices))

train_idx, val_idx = folds[fold_idx]
train_pairs = [train_val_pairs[i] for i in train_idx]
val_pairs = [train_val_pairs[i] for i in val_idx]



all_train_scores = []
all_val_scores = []
all_test_scores = []

print(f"\n📦 Fold {fold_idx + 1}/{K}: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
fold_dir = os.path.join(save_root, f"fold{fold_idx + 1}")
os.makedirs(fold_dir, exist_ok=True)

save_fold_split_index(train_pairs, val_pairs, test_pairs, fold_idx, save_dir=fold_dir)

train_pairs = [train_val_pairs[i] for i in train_idx]
val_pairs = [train_val_pairs[i] for i in val_idx]

# 기본 transform
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB 정규화
])

train_dataset = SegDataset(train_pairs, transform=transform)
val_dataset = SegDataset(val_pairs, transform=transform)
test_dataset = SegDataset(test_pairs, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# 모델 초기화
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)
model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

# 전략 1 : loss 변경 (dice -> dice+bce)
bce = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode='binary')
focal = smp.losses.FocalLoss(mode='binary', gamma=2.0)
def loss_fn(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

# 학습 루프
train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []

best_val_dice = 0.0
best_model_path = fold_dir+f"/best_model_fold{fold_idx}.pt"
for epoch in range(num_epochs):
    # === Train ===
    model.train()
    running_loss = 0.0
    train_dice = 0.0
    for images, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        t_dice = dice_score(torch.sigmoid(preds), masks)
        train_dice += t_dice.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    train_losses.append(avg_train_loss)
    train_dice_scores.append(avg_train_dice)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)
            val_loss += loss.item()
            v_dice = dice_score(torch.sigmoid(preds), masks)
            val_dice += v_dice.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)

    # 매 epoch 끝에 loss 기준으로 조절
    scheduler.step(avg_val_loss)

    print(
        f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f} | Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_dice": avg_val_dice
        }, best_model_path)
        print(f"📌 [Epoch {epoch + 1}] 🥇 New best val Dice: {avg_val_dice:.4f} → 모델 저장됨!")

# loss curve 저장
save_loss_curve(
    train_losses, val_losses, train_dice_scores, val_dice_scores,
    save_dir=fold_dir
)

# === Test Evaluation ===
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint["model_state_dict"])
best_epoch = checkpoint["epoch"]
best_val_dice = checkpoint["val_dice"]

print(f"✅ Loaded best model from epoch {best_epoch} (val Dice: {best_val_dice:.4f})")

model.eval()
test_dice = 0.0

for images, masks in test_loader:
    for i in range(images.size(0)):
        pred = torch.sigmoid(model(images[i].unsqueeze(0).to(device)))
        mask = masks[i].unsqueeze(0).to(device)
        score = dice_score(pred, mask)
        test_dice += score.item()
        count += 1
avg_test_dice = test_dice / count

print(f"\n🎯 Final Test Dice Score: {avg_test_dice:.4f}")

# 평가 결과 저장
train_df = evaluate_and_save_metrics(train_loader, model, device, split_name=f"train_fold{fold_idx + 1}",
                                   save_dir=fold_dir)
val_df = evaluate_and_save_metrics(val_loader, model, device, split_name=f"val_fold{fold_idx + 1}",
                                   save_dir=fold_dir)
test_df = evaluate_and_save_metrics(test_loader, model, device, split_name=f"test_fold{fold_idx + 1}",
                                    save_dir=fold_dir)

# 시각화 저장
visualize_and_save_predictions(train_loader, model, device, split_name=f"train_fold{fold_idx + 1}",
                               save_dir=fold_dir)
visualize_and_save_predictions(val_loader, model, device, split_name=f"valid_fold{fold_idx + 1}",
                               save_dir=fold_dir)
visualize_and_save_predictions(test_loader, model, device, split_name=f"test_fold{fold_idx + 1}",
                               save_dir=fold_dir)

log_fold_score(train_df, val_df, test_df, log_path=save_root+'/fold_scores.csv', fold=fold_idx)
show_cv_summary_if_complete(csv_path=save_root+'/fold_scores.csv', expected_folds=5)
