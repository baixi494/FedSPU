import torch
from torch import nn
import numpy as np

class Utils:
    @staticmethod
    def evaluate(testloader, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.0 * correct / total

    @staticmethod
    def sd_to_vec(sd: dict) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in sd.values()])

    @staticmethod
    def vec_to_sd(vec: torch.Tensor, tmpl: nn.Module) -> dict:
        out = {}
        offset = 0
        for name, p in tmpl.state_dict().items():
            num = p.numel()
            out[name] = vec[offset:offset+num].view_as(p).clone()
            offset += num
        return out

    @staticmethod
    def l2_project(curr: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
        diff = curr - center
        n = torch.norm(diff)
        if n > radius and n > 0:
            diff = diff * (radius / n)
        return center + diff

    @staticmethod
    def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.dot(a, b)

    @staticmethod
    def quad_gain(delta: torch.Tensor, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return Utils.dot(g, delta) + 0.5 * Utils.dot(h * delta, delta)

from torch.utils.data import Dataset, DataLoader

class _ASRStrictDataset(Dataset):
    def __init__(self, base_ds, target_label, apply_trigger_fn):

        self.items = [(img, lbl) for (img, lbl) in base_ds if int(lbl) != int(target_label)]
        self.tau = int(target_label)
        self.apply_trigger = apply_trigger_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, _ = self.items[idx]
        img_t = self.apply_trigger(img)
        return img_t, self.tau

def make_asr_strict_loader(test_dataset, target_label, apply_trigger_fn, batch_size=128, num_workers=2):
    ds = _ASRStrictDataset(test_dataset, target_label, apply_trigger_fn)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def evaluate_asr_strict(asr_loader, model) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    hit, total = 0, 0
    with torch.no_grad():
        for images, labels in asr_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            hit += (pred == labels).sum().item()
            total += labels.numel()
    return 100.0 * hit / max(total, 1)
