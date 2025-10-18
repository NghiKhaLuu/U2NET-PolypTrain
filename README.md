# Project Structure  
<img width="313" height="511" alt="image" src="https://github.com/user-attachments/assets/bc005d96-0e73-425f-af44-e87ecbaac36f" />  
  
# Requirements  
- Python 3.10+  
- PyTorch 2.2+  
- OpenCV, Pillow, albumentations
  
# Setup  
python -m venv venv  
venv\Scripts\activate  
pip install opencv-python pillow tqdm albumentations  
  
If using GPU (>= NVIDIA GeForce RTX 2050):  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
If using CPU:  
pip install torch torchvision torchaudio  
  
# Prepare the dataset  
Place your dataset (including both frames and masks) into the dataset folder.  
Recommended dataset: KVarsir Segmentation Dataset - https://www.kaggle.com/datasets/kuece16/kvarsir-seg/data  
  
# Install U2Net  
Create a folder U2NET/ in your project.  
Download u2net.py from the official repo: https://github.com/xuebinqin/U-2-Net.  
Copy it into the U2NET/ folder.  
Alternatively, you can use the entire U2NET folder from U2NET-PolypTrain.  
  
# Create train_u2net.py Or download it from U2NET-PolypTrain.  
  
                     #Example code structure:
                     import os, cv2, torch, torch.nn as nn
                     import torch.optim as optim
                     from torch.utils.data import Dataset, DataLoader, random_split
                     from tqdm import tqdm
                     from U2NET.u2net import U2NET
                     
                     #Dataset class
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
                     
                     # Loss functions
                     def dice_loss(pred, target, smooth=1e-6):
                         intersection = (pred*target).sum()
                         return 1 - (2*intersection + smooth)/(pred.sum()+target.sum()+smooth)
                     
                     bce_loss_fn = nn.BCEWithLogitsLoss()
                     
                     def u2net_loss(d_outputs, masks):
                         loss = 0
                         for d in d_outputs:
                             loss += bce_loss_fn(d, masks) + dice_loss(torch.sigmoid(d), masks)
                         return loss
                     
                     # Training setup
                     device = 'cuda' if torch.cuda.is_available() else 'cpu'
                     print("Using device:", device)
                     
                     dataset = PolypDataset('dataset/images', 'dataset/masks', size=320)
                     train_size = int(0.9*len(dataset))
                     val_size = len(dataset)-train_size
                     train_ds, val_ds = random_split(dataset, [train_size, val_size])
                     
                     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
                     val_loader = DataLoader(val_ds, batch_size=4)
                     
                     model = U2NET(3,1).to(device)
                     optimizer = optim.Adam(model.parameters(), lr=1e-4)
                     epochs = 50
                     
                     # Mixed precision
                     scaler = torch.cuda.amp.GradScaler()
                     
                     # Checkpoint folder
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
                         
                         # Save checkpoint every 5 epochs
                         if (epoch+1) % 5 == 0:
                             ckpt_path = f"saved_models/u2net_epoch{epoch+1}_320px.pth"
                             torch.save(model.state_dict(), ckpt_path)
                             print(f"Saved checkpoint: {ckpt_path}")

# Run the training:  
venv\Scripts\activate  
python train_u2net.py  
  
# If you want to use a pre-trained .pth file  
Download the checkpoint: u2net_epoch50_320px.pth - https://drive.google.com/drive/folders/11I0q6opspbhRh3rnrGmbqeVhHcGpf0bK?usp=drive_link  
In your project folder, after cloning U2NET:  git clone https://github.com/xuebinqin/U-2-Net.git U2NET  
Place the .pth file here:  
U2NET/saved_models/u2net/u2net_epoch50_320px.pth  
  
         #Load it in your script for frame-to-mask conversion:  
         model_path = os.path.normpath(os.path.join(base_dir, "../U2NET/saved_models/u2net/u2net_epoch50_320px.pth"))
         print("Loading U2NET model ...")
         net = U2NET(3, 1)
         state_dict = torch.load(model_path, map_location=device)
         net.load_state_dict(state_dict)
         net.to(device)
         net.eval()
         print("[OK] Model loaded successfully!\n")
  
Additional reference: VIDEO-POLYP-SEG project (links will be updated as training progresses).  
  
# Example:  
Before:  
<img width="832" height="360" alt="image" src="https://github.com/user-attachments/assets/e6accceb-d38d-4ad5-a024-98766a5ad56f" />  
After:  
<img width="832" height="363" alt="image" src="https://github.com/user-attachments/assets/0687a8a7-04d7-4b4b-8e6c-cab69c806f1b" />  
  
# Author  
Name: Luu Kha Nghi  
University: Can Tho University  - VietNam  
Course: CT255 - Intelligent Business Operations 
GitHub: NghiKhaLuu  
