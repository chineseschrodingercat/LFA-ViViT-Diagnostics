import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VideoClassificationDataset(Dataset):
    """
    Custom Dataset for LFA Video Classification using ViViT.
    Handles video frame extraction, temporal sampling, and normalization.
    """
    def __init__(self, root_dir, cutoff=10, num_frames=32, prefix_frames=None, transform=None):
        self.samples = []
        self.cutoff = cutoff
        self.num_frames = num_frames
        self.prefix_frames = prefix_frames
        
        # Standard ImageNet normalization for ViViT pre-trained models
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Scan directory for video files
        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist.")
        else:
            for conc in os.listdir(root_dir):
                try:
                    conc_val = float(conc)
                except ValueError:
                    continue
                
                # Labeling logic: 1 if concentration >= cutoff, else 0
                label = 1.0 if conc_val >= cutoff else 0.0
                conc_path = os.path.join(root_dir, conc)
                
                for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
                    for vid in glob.glob(os.path.join(conc_path, ext)):
                        self.samples.append((vid, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            # Fallback for corrupted videos: return black frames
            print(f"Warning: Could not open video {path}")
            return torch.zeros((self.num_frames, 3, 224, 224)), torch.tensor([label], dtype=torch.float32)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Temporal cropping logic: Focus on the initial reaction phase (prefix)
        if self.prefix_frames is not None:
            P = min(total_frames, self.prefix_frames)
        else:
            P = total_frames

        if P <= 0:
            idxs = np.zeros(self.num_frames, dtype=int)
        elif self.num_frames == 1:
            idxs = np.array([P - 1])
        else:
            # Uniform sampling strategy
            if P >= self.num_frames:
                idxs = np.linspace(0, P - 1, self.num_frames).astype(int)
            else:
                # Padding strategy: Repeat the last frame if video is too short
                pad_len = self.num_frames - P
                idxs = np.concatenate([
                    np.arange(P),
                    np.full(pad_len, P - 1)
                ])

        # Ensure indices are within bounds
        idxs = np.clip(idxs, 0, total_frames - 1)

        frames = []
        current_frame_idx = -1

        # Optimized frame seeking and decoding
        for frame_idx in idxs:
            if frame_idx != current_frame_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            current_frame_idx = frame_idx

            if ret:
                # OpenCV loads in BGR, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = self.transform(frame)
                frames.append(img)
            else:
                # Handle read errors by repeating previous frame or black frame
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros((3, 224, 224)))

        cap.release()

        if len(frames) > 0:
            final_tensor = torch.stack(frames)
        else:
            final_tensor = torch.zeros((self.num_frames, 3, 224, 224))

        return final_tensor, torch.tensor([label], dtype=torch.float32)
