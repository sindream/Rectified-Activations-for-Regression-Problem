import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from layers import *
from resnet_encoder_react import ResnetEncoder
from depth_decoder import DepthDecoder
import pandas as pd
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

class DIODEDataset(Dataset):
    def __init__(self, df, image_transform=None, depth_transform=None):
        self.df = df
        self.image_transform = image_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.df.iloc[idx]["image"]
        img = Image.open(img_path).convert('RGB')

        # 깊이 맵 로드
        depth_path = self.df.iloc[idx]["depth"]
        depth = np.load(depth_path).astype('float32')

        # 마스크 로드
        mask_path = self.df.iloc[idx]["mask"]
        mask = np.load(mask_path).astype(bool)

        # 불필요한 차원 제거
        depth = np.squeeze(depth)
        mask = np.squeeze(mask)

        # 마스크 적용
        depth = np.where(mask, depth, 0.0)

        # 깊이 맵을 PIL 이미지로 변환
        depth_img = Image.fromarray(depth)

        # 변환 적용
        if self.image_transform:
            img = self.image_transform(img)
        if self.depth_transform:
            depth = self.depth_transform(depth_img)
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()

        return img, depth

if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    annotation_folder = "/dataset/"

    if not os.path.exists(os.path.abspath(".") + annotation_folder):
        import tensorflow as tf
        from tensorflow import keras
        annotation_zip = keras.utils.get_file(
            "val.tar.gz",
            cache_subdir=os.path.abspath("."),
            origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
            extract=True,
        )

    def get_file_list(path):
        filelist = []
        for root, dirs, files in os.walk(path):
            for file in files:
                filelist.append(os.path.join(root, file))
        filelist.sort()
        return filelist
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    path_outdoor = "./val/outdoor"

    filelist_outdoor = get_file_list(path_outdoor)
    data_outdoor = {
        "image": [x for x in filelist_outdoor if x.endswith(".png")],
        "depth": [x for x in filelist_outdoor if x.endswith("_depth.npy")],
        "mask": [x for x in filelist_outdoor if x.endswith("_depth_mask.npy")],
    }

    df_outdoor = pd.DataFrame(data_outdoor)
    df_outdoor = df_outdoor.sample(frac=1, random_state=42).reset_index(drop=True)
    train_ratio = 0.99
    train_size = int(len(df_outdoor) * train_ratio)
    val_size = len(df_outdoor) - train_size

    df_train = df_outdoor.iloc[:train_size].reset_index(drop=True)
    df_val = df_outdoor.iloc[train_size:].reset_index(drop=True)

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")

    # 변환 정의
    image_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.ToTensor(),
    ])

    depth_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.ToTensor(),
    ])


    # DataLoader 설정
    batch_size = 1

    # Train DataLoader
    dataset_train = DIODEDataset(df_train, image_transform=image_transform, depth_transform=depth_transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    # 모델 초기화
    num_layers = 18
    model = MonoDepth2Model(num_layers).to(device)
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

            _, pred_depth = disp_to_depth(disp_resized, 0.3, 300.0)

            pred_depth_full = pred_depth.clone()
            gt_depth_full = gt_depth.clone()

            valid_mask = (gt_depth > 0)
            pred_depth = pred_depth[valid_mask]
            gt_depth = gt_depth[valid_mask]

            if pred_depth.numel() > 0:
                # 값 클램핑 (클램핑을 먼저 수행)
                pred_depth = torch.clamp(pred_depth, min=0.3, max=300)
                gt_depth = torch.clamp(gt_depth, min=0.3, max=300)

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

            if idx == 35:
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