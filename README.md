Cấu trúc thư mục:  
<img width="313" height="511" alt="image" src="https://github.com/user-attachments/assets/bc005d96-0e73-425f-af44-e87ecbaac36f" />  
1. Trường hợp muốn tự train lại  
   Bước 1: Chuẩn bị môi trường  
      python -m venv venv  
      venv\Scripts\activate  
      pip install opencv-python pillow tqdm albumentations  
      Nếu dùng GPU (>=NVIDIA GeForce RTX 2050): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
      Nếu dùng CPU: pip install torch torchvision torchaudio  
   Bước 2: Chuẩn bị dataset  
       Đưa dataset mà bạn muốn dùng đề huấn kuyện vào thư mục dataset bao gồm cả frames và mask.  
       Đề xuất lấy tại: https://www.kaggle.com/datasets/kuece16/kvarsir-seg/data  
   Bước 3: Bước 3: Cài U2Net  
      Tạo folder U2NET/ trong dự án.  
      Download file u2net.py từ repo chính: https://github.com/xuebinqin/U-2-Net.  
      Copy vào folder U2NET/.  
      Hoặc lấy cả folder U2NET tại U2NET-PolypTrain.  
   Bước 4: Tạo file train_u2net.py hoặc tải về từ U2NET-PolypTrain.  
              import os, cv2, torch, torch.nn as nn  
              import torch.optim as optim  
              from torch.utils.data import Dataset, DataLoader, random_split  
              from tqdm import tqdm  
              from U2NET.u2net import U2NET  
                
              # Dataset  
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
              
              # Loss
              def dice_loss(pred, target, smooth=1e-6):
                  intersection = (pred*target).sum()
                  return 1 - (2*intersection + smooth)/(pred.sum()+target.sum()+smooth)
              
              bce_loss_fn = nn.BCEWithLogitsLoss()  # khởi tạo 1 lần ngoài vòng lặp
              def u2net_loss(d_outputs, masks):
                  loss = 0
                  for d in d_outputs:
                      loss += bce_loss_fn(d, masks) + dice_loss(torch.sigmoid(d), masks)
                  return loss
              
              
              # Training
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
   Bước 5: Ở venv chạy python train_u2net.py  
  
2. Trường hợp muốn dùng file .pth đã qua train:  
  Tải file u2net_epoch50_320px.pth tại: https://drive.google.com/drive/folders/11I0q6opspbhRh3rnrGmbqeVhHcGpf0bK?usp=drive_link  
  Tại thư mục dự án của bạn, sau khi clone U2NET (git clone https://github.com/xuebinqin/U-2-Net.git U2NET)  
  Trong U2NET  
    =>saved_models  
    	=>Tạo folder: u2net  
    		=>Đưa file .pth vào: u2net_epoch50_320px.pth  
   (U2NET/saved_models/u2net/u2net_epoch50_320px.pth)  
   Trong file .py dùng để chuyển frames=>masks:  
     # Đường dẫn đến model U2NET  
        model_path = os.path.normpath(os.path.join(base_dir, "../U2NET/saved_models/u2net/u2net_epoch50_320px.pth"))
     # Load model  
        print("Loading U2NET model ...")
        net = U2NET(3, 1)
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        print("[OKE] Model loaded successfully!\n")
   Tham khảo thêm tại VIDEO-POLYP-SEG (sẽ cập nhật thêm đường dẫn sau vì dự án vẫn còn đang train)  
3. Author:  
      Name	          Lưu Khả Nghị  
      University	    Đại học Cần Thơ  
      Course	        CT255 – Nghiệp vụ Thông minh  
      GitHub	        NghiKhaLuu  
