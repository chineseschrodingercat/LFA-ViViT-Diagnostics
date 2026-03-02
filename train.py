import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import VivitForVideoClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import VideoClassificationDataset  # Import our clean dataset class

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = "./data/train"  # Update this path
VAL_DIR = "./data/val"      # Update this path
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
MODEL_NAME = "google/vivit-b-16x2-kinetics400"

# Hyperparameters
NUM_EPOCHS = 30
NUM_FRAMES = 32
CUTOFF = 10
BATCH_SIZE = 8
PREFIX_FRAMES = 3600
LR = 1e-4

def main():
    print(f"Using device: {DEVICE}")

    # Initialize Datasets
    train_dataset = VideoClassificationDataset(TRAIN_DIR, cutoff=CUTOFF, num_frames=NUM_FRAMES, prefix_frames=PREFIX_FRAMES)
    val_dataset = VideoClassificationDataset(VAL_DIR, cutoff=CUTOFF, num_frames=NUM_FRAMES, prefix_frames=PREFIX_FRAMES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load Pretrained ViViT Model
    print("Loading ViViT model...")
    model = VivitForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1, 
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        for videos, labels in train_loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1)

            outputs = model(pixel_values=videos, interpolate_pos_encoding=True)
            logits = outputs.logits.view(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        scheduler.step()
        avg_train_loss = total_train_loss / total_samples

        # Validation Step
        model.eval()
        correct = 0
        total_val = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1)

                outputs = model(pixel_values=videos)
                logits = outputs.logits.view(-1)
                preds = (logits > 0).long()

                correct += (preds == labels.long()).sum().item()
                total_val += labels.size(0)

        acc = correct / total_val
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f}")

        # Save Checkpoint
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"*** New Best Model Saved (Acc: {best_val_acc:.4f}) ***")

if __name__ == "__main__":
    main()
