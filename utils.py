import torch
import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import KFold

def img_load(img_dir, mask_dir):
    # 파일 매칭: 'benign_Adjusted_123.png' ↔ 'pixelLabel_123.png'
    img_files = sorted(list(img_dir.glob("*.png")))
    mask_files = sorted(list(mask_dir.glob("*.png")))

    def extract_orig_img_idx(f):
        # 괄호 속 숫자 (예: benign_Adjusted (12).png → "12")
        match = re.search(r"\((\d+)\)", f.stem)
        return match.group(1).zfill(3) if match else None  # "12" → "012"

    def extract_fuzzy_img_idx(f):
        # 파일명에서 뒤의 숫자 추출 (예: pixelLabel_001.png → "001")
        match = re.search(r"_(\d+)", f.stem)
        return match.group(1) if match else None

    def extract_mask_idx(f):
        # 파일명에서 뒤의 숫자 추출 (예: pixelLabel_001.png → "001")
        match = re.search(r"_(\d+)", f.stem)
        return match.group(1) if match else None

    if 'Fuzzy' in str(img_dir):
        paired_files = [
            (img, next(m for m in mask_files if extract_mask_idx(m) == extract_fuzzy_img_idx(img)))
            for img in img_files
            if extract_fuzzy_img_idx(img) and any(extract_mask_idx(m) == extract_fuzzy_img_idx(img) for m in mask_files)
        ]
    else:
        paired_files = [
            (img, next(m for m in mask_files if extract_mask_idx(m) == extract_orig_img_idx(img)))
            for img in img_files
            if extract_orig_img_idx(img) and any(extract_mask_idx(m) == extract_orig_img_idx(img) for m in mask_files)
        ]

    return paired_files

def split_fixed_test(paired_files, test_ratio=0.1, seed=42):
    N = len(paired_files)
    indices = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(indices)

    test_size = int(N * test_ratio)
    test_idx = indices[:test_size]
    train_val_idx = indices[test_size:]

    test_pairs = [paired_files[i] for i in test_idx]
    train_val_pairs = [paired_files[i] for i in train_val_idx]

    return train_val_pairs, test_pairs

def extract_img_idx_from_name(path):
    """파일명에서 괄호 안 숫자 추출 → int로 리턴"""
    match = re.search(r"\((\d+)\)", path.stem)
    return int(match.group(1)) if match else -1

def save_fold_split_index(train_pairs, val_pairs, test_pairs, fold_idx, save_dir="./results"):
    all_data = []

    def add_split_data(pairs, split_name):
        for img_path, mask_path in pairs:
            index = extract_img_idx_from_name(img_path)
            all_data.append({
                "index": index,
                "split": split_name,
                "img_path": str(img_path),
                "mask_path": str(mask_path)
            })

    add_split_data(train_pairs, "train")
    add_split_data(val_pairs, "val")
    add_split_data(test_pairs, "test")

    df = pd.DataFrame(all_data)
    df = df.sort_values(["split", "index"]).reset_index(drop=True)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"fold{fold_idx+1}_split_index.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Split index for fold {fold_idx} saved to: {out_path}")


def evaluate_and_save_metrics(loader, model, device, split_name="val", save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            preds = model(images).sigmoid()
            preds_bin = (preds > 0.5).float()

            for i in range(images.size(0)):
                dice = dice_score(preds_bin[i], masks[i]).item()
                results.append({
                    "split": split_name,
                    "index": batch_idx * loader.batch_size + i,
                    "dice_score": round(dice, 4)
                })

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, f"{split_name}_dice_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {split_name} dice scores to: {csv_path}")
    return df

def visualize_and_save_predictions(loader, model, device, split_name="val", save_dir="./results", threshold=0.5):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    vis_dir = os.path.join(save_dir, split_name)
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images).sigmoid().cpu()
            preds_bin = (preds > threshold).float()

            for i in range(images.size(0)):
                idx = batch_idx * loader.batch_size + i

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img + 1.0) / 2.0  # ✅ [-1,1] → [0,1]로 복원
                img = np.clip(img, 0, 1)  # ✅ 혹시 모를 out-of-bound 제거
                gt = masks[i].cpu().squeeze().numpy()
                pred = preds[i].squeeze().numpy()
                pred_bin = preds_bin[i].squeeze().numpy()

                # Dice 계산
                intersection = (pred_bin * gt).sum()
                union = pred_bin.sum() + gt.sum()
                smooth = 1e-5
                dice = (2. * intersection + smooth) / (union + smooth)

                # 시각화
                fig, ax = plt.subplots(1, 3, figsize=(8, 3))
                ax[0].imshow(img)
                ax[0].set_title("Input")
                ax[1].imshow(gt, cmap="gray")
                ax[1].set_title("GT")
                ax[2].imshow(pred_bin, cmap="gray")
                ax[2].set_title(f"Pred (Dice: {dice:.3f})")

                for a in ax:
                    a.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"sample_{idx:03d}.png"))
                plt.close()

    print(f"✅ Saved {split_name} visualizations to: {vis_dir}")



def save_loss_curve(train_losses, val_losses, train_dices, val_dices, save_dir):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    save_path = save_dir + '/loss_curve.png'
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Dice Curve
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice', marker='o')
    plt.plot(val_dices, label='Val Dice', marker='s')
    plt.title("Dice Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved loss curve to {save_path}")

def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    smooth = 1e-5
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def log_fold_score(train_df, val_df, test_df, fold, log_path):
    score_row = {
        "fold": fold,
        "train_dice": train_df["dice_score"].mean(),
        "val_dice": val_df["dice_score"].mean(),
        "test_dice": test_df["dice_score"].mean()
    }

    # CSV가 이미 있다면 append, 없으면 새로 생성
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        existing = existing[existing["fold"] != fold]  # 덮어쓰기 방지
        df = pd.concat([existing, pd.DataFrame([score_row])], ignore_index=True)
    else:
        df = pd.DataFrame([score_row])

    df = df.sort_values("fold").reset_index(drop=True)
    df.to_csv(log_path, index=False)
    print(f"✅ Fold {fold} score logged to {log_path}")

import pandas as pd

def show_cv_summary_if_complete(csv_path, expected_folds=5):
    if not os.path.exists(csv_path):
        print("❌ fold_scores.csv not found.")
        return

    df = pd.read_csv(csv_path)
    if len(df) < expected_folds:
        print(f"📌 현재까지 {len(df)}개의 fold 결과만 존재합니다 (총 {expected_folds}개 필요).")
        return

    print("\n📊 ✅ 5-Fold Cross-Validation Summary")
    for col in ["train_dice", "val_dice", "test_dice"]:
        mean = df[col].mean()
        std = df[col].std()
        print(f"{col}: {mean:.4f} ± {std:.4f}")

    print("\n📁 상세 결과:")
    print(df)

import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        TP = (pred * target).sum(dim=(1, 2, 3))
        FP = (pred * (1 - target)).sum(dim=(1, 2, 3))
        FN = ((1 - pred) * target).sum(dim=(1, 2, 3))
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1. - tversky.mean()
