import torch
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from .model import UNET
from albumentations.pytorch import ToTensorV2

#from utils import( load_checkpoint , save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

#hyperparameters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../Dataset/train"
VAL_IMG_DIR  = "../Dataset/train"
TRAIN_MASK_DIR = "../Dataset/train_masks"
VAL_MASK_DIR = "../Dataset/train_masks"


def train_fn(loader, model, optimizer, loss_fn , scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            pass
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss = loss.item())


if __name__ == "__main__":
    train_transform = A.compose(
        [
            A.resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.rotate(limit = 35.0, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.Normalize(
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    max_pixel_value = 255.0
            ),
            ToTensorV2(),
        ],
    )
    
    val_transform = A.Compose([
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    max_pixel_value = 255.0
            ),
            ToTensorV2()
        
    ])
    
    model  = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()# constructor for a callable object
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_loaders(
        
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE, 
        train_transform,
        val_transform,
        NUM_WORKERS, 
        PIN_MEMORY
        )
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        