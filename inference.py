import torch
import argparse
from transformers import VivitForVideoClassification
from dataset import VideoClassificationDataset
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference(model_path, test_dir):
    # Load Model Structure
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # Load Weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Prepare Data
    test_ds = VideoClassificationDataset(test_dir, cutoff=10, num_frames=32, prefix_frames=3600)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    print("Starting Inference...")
    results = []
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(DEVICE)
            logits = model(pixel_values=videos).logits.view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            # Store results (simplified)
            results.extend(preds.cpu().numpy())

    print(f"Inference completed. Processed {len(results)} videos.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test videos")
    args = parser.parse_args()

    run_inference(args.weights, args.test_dir)
