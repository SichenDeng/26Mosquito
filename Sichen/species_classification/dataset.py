"""Dataset + transforms for mosquito species classification.

A PyTorch Dataset answers two questions: "how many samples?" (__len__) and
"give me sample i" (__getitem__ -> (image tensor, label)). The DataLoader then
batches and shuffles them for training.
"""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "processed" / "ong2021_800"
SPLITS_CSV = ROOT / "data" / "splits.csv"

# Fixed class order: the model's output 0 = aegypti, 1 = albopictus. Saved with the model later.
CLASSES = ["aegypti", "albopictus"]

INPUT_SIZE = (384, 512)  # (height, width): keeps the 4:3 ratio instead of squashing to a square
# The pretrained ResNet was trained on ImageNet images normalized with these values,
# so our images must be normalized the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train):
    if train:
        # Augmentation = random changes that don't change the species, so the model sees
        # more variety. Flips are safe (orientation means nothing here); horizontal + vertical
        # flip together = 180 degree rotation. Color changes are kept mild so the white
        # scale patterns are not washed out.
        return transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return eval_transform(INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD)


def eval_transform(input_size, mean, std):
    """Validation / test / prediction: no randomness, so results are repeatable.
    evaluate.py and predict.py call this with the values stored in the checkpoint,
    so a saved model is always fed images prepared exactly as during its training."""
    return transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class MosquitoDataset(Dataset):
    def __init__(self, split, transform=None, image_dir=IMAGE_DIR):
        with open(SPLITS_CSV) as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(self.image_dir / row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASSES.index(row["species"])
        return image, label
