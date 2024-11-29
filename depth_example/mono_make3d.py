import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from layers import *
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder
import scipy.io
import os
import matplotlib.pyplot as plt

class MonoDepth2Model(nn.Module):
    def __init__(self, num_layers):
        super(MonoDepth2Model, self).__init__()
        self.encoder = ResnetEncoder(18, False)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    def forward(self, input_image):
        features = self.encoder(input_image)
        output = self.decoder(features)
        return output

class Make3DDataset(Dataset):
    def __init__(self, image_dir, depth_dir, image_transform=None, depth_transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        
        # 이미지와 깊이 파일 경로 정렬
        self.image_files = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".jpg")])
        self.depth_files = sorted([os.path.join(depth_dir, file) for file in os.listdir(depth_dir) if file.endswith(".mat")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        # 깊이 맵 로드 (MAT 파일에서 'Position3DGrid' 키 사용)
        depth_path = self.depth_files[idx]
        depth_data = scipy.io.loadmat(depth_path)
        depth = depth_data["Position3DGrid"][:, :, -1].astype('float32')  # 깊이 정보는 세 번째 축에 위치합니다

        # 변환 적용
        if self.image_transform:
            img = self.image_transform(img)
        if self.depth_transform:
            depth_img = Image.fromarray(depth)
            depth = self.depth_transform(depth_img)
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()

        return img, depth

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 경로 설정
    image_dir = "make3d/image"
    depth_dir = "make3d/depth/Train400Depth"

    # 변환 정의
    image_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.ToTensor(),
    ])

    depth_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.ToTensor(),
    ])

    # 데이터셋 준비
    dataset = Make3DDataset(image_dir=image_dir, depth_dir=depth_dir, image_transform=image_transform, depth_transform=depth_transform)
    
    # 데이터 분할
    train_ratio = 1.0
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # DataLoader 설정
    batch_size = 1
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 초기화
    num_layers = 18
    model = MonoDepth2Model(num_layers).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # 가중치 로드
    encoder_path = 'model_nopt/encoder.pth'
    decoder_path = 'model_nopt/depth.pth'

    # 인코더 가중치 로드
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in model.encoder.state_dict()}
    model.encoder.load_state_dict(filtered_dict_enc)

    # 디코더 가중치 로드
    loaded_dict_dec = torch.load(decoder_path, map_location=device)
    filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in model.decoder.state_dict()}
    model.decoder.load_state_dict(filtered_dict_dec)

    model.eval()
    # 평가 지표 초기화
    abs_rel = 0.0
    sq_rel = 0.0
    rmse = 0.0
    rmse_log = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    num_samples = 0

    with torch.no_grad():
        for idx, (input_image, gt_depth) in enumerate(loader_train):
            
            input_image = input_image.to(device)
            gt_depth = gt_depth.to(device)
            
            outputs = model(input_image)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, size=gt_depth.shape[-2:], mode='bilinear', align_corners=False
            )

            _, pred_depth = disp_to_depth(disp_resized, 0.5, 81.0)

            pred_depth_full = pred_depth.clone()
            gt_depth_full = gt_depth.clone()

            valid_mask = (gt_depth > 0)
            pred_depth = pred_depth[valid_mask]
            gt_depth = gt_depth[valid_mask]

            if pred_depth.numel() > 0:
                # 값 클램핑 (클램핑을 먼저 수행)
                pred_depth = torch.clamp(pred_depth, min=0.5, max=81)
                gt_depth = torch.clamp(gt_depth, min=0.5, max=81)

                # Median Scaling 적용
                scale = torch.median(gt_depth) / torch.median(pred_depth)
                pred_depth *= scale
                pred_depth_full *= scale

                # 지표 계산
                abs_rel += torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth).item()
                sq_rel += torch.mean(((gt_depth - pred_depth) ** 2) / gt_depth).item()
                rmse += torch.sqrt(torch.mean((gt_depth - pred_depth) ** 2)).item()
                rmse_log += torch.sqrt(torch.mean((torch.log(gt_depth) - torch.log(pred_depth)) ** 2)).item()

                thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
                a1 += (thresh < 1.25).float().mean().item()
                a2 += (thresh < 1.25 ** 2).float().mean().item()
                a3 += (thresh < 1.25 ** 3).float().mean().item()

                num_samples += 1

            if idx == 15:
                pred_depth_np = pred_depth_full[0, 0].cpu().numpy()
                gt_depth_np = gt_depth_full[0, 0].cpu().numpy()
                valid_mask_np = valid_mask[0, 0].cpu().numpy()

                # 유효하지 않은 영역을 0으로 설정
                pred_depth_np[~valid_mask_np] = 0
                gt_depth_np[~valid_mask_np] = 0

                # 시각화를 위해 깊이 맵을 정규화
                vmax = np.percentile(gt_depth_np[valid_mask_np], 95)

                plt.figure(figsize=(12, 5))

                plt.subplot(1, 3, 1)
                plt.title('Input Image')
                input_img_np = input_image[0].cpu().numpy().transpose(1, 2, 0)
                plt.imshow(input_img_np)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title('Ground Truth Depth')
                plt.imshow(gt_depth_np, cmap='magma', vmax=vmax)
                plt.colorbar()
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title('Predicted Depth')
                plt.imshow(pred_depth_np, cmap='magma', vmax=vmax)
                plt.colorbar()
                plt.axis('off')

                plt.tight_layout()
                plt.show()
                
    # 평균 지표 계산
    if num_samples > 0:
        abs_rel /= num_samples
        sq_rel /= num_samples
        rmse /= num_samples
        rmse_log /= num_samples
        a1 /= num_samples
        a2 /= num_samples
        a3 /= num_samples

        print(f"\nValidation Results:")
        print(f"  Abs Rel: {abs_rel:.4f}")
        print(f"  Sq Rel: {sq_rel:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  RMSE log: {rmse_log:.4f}")
        print(f"  a1: {a1:.4f}")
        print(f"  a2: {a2:.4f}")
        print(f"  a3: {a3:.4f}")
    else:
        print("No valid samples found for evaluation.")