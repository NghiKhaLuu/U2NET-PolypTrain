import os, cv2, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from U2NET.u2net import U2NET

# --------- 1. Dataset ----------
class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=320):
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        image = torch.from_numpy(image/255.0).float().permute(2,0,1)
        mask = torch.from_numpy(mask/255.0).float().unsqueeze(0)
        return image, mask

# --------- 2. Loss ----------
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred*target).sum()
    return 1 - (2*intersection + smooth)/(pred.sum()+target.sum()+smooth)

bce_loss_fn = nn.BCEWithLogitsLoss()  # khởi tạo 1 lần ngoài vòng lặp
def u2net_loss(d_outputs, masks):
    loss = 0
    for d in d_outputs:
        loss += bce_loss_fn(d, masks) + dice_loss(torch.sigmoid(d), masks)
    return loss


# --------- 3. Training ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Dataset
dataset = PolypDataset('dataset/images', 'dataset/masks', size=320)
train_size = int(0.9*len(dataset))
val_size = len(dataset)-train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# Model & optimizer
model = U2NET(3,1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 50

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# Folder checkpoint
os.makedirs("saved_models", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            d_outputs = model(imgs)
            loss = u2net_loss(d_outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(train_loss=total_loss/len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            d_outputs = model(imgs)
            val_loss += u2net_loss(d_outputs, masks).item()
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save checkpoint mỗi 5 epoch
    if (epoch+1) % 5 == 0:
        ckpt_path = f"saved_models/u2net_epoch{epoch+1}_320px.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        
# venv\Scripts\activate
# python train_u2net.py