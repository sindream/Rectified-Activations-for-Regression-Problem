import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io  # For loading .mat files
import cv2  # For loading image files

def load_diode_dataset(data_dir):
    """
    Load depth data from the DIODE dataset stored in .npy files.
    """
    depth_list = []
    # Traverse through the dataset
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_depth.npy'):
                depth_path = os.path.join(root, file)
                mask_path = depth_path.replace('_depth.npy', '_depth_mask.npy')
                # Load depth and mask
                depth = np.load(depth_path).astype('float32').squeeze()
                mask = np.load(mask_path).astype(bool).squeeze()
                # Apply mask
                depth = np.where(mask, depth, 0.0)
                depth_list.append(depth.flatten())
    if depth_list:
        depth_data = np.concatenate(depth_list)
        return depth_data
    else:
        return np.array([])

def load_make3d_dataset(image_dir, depth_dir):
    """
    Load depth data from the Make3D dataset stored in .mat files.
    """
    depth_list = []
    # Get sorted lists of image and depth files to ensure alignment
    image_files = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')])
    depth_files = sorted([os.path.join(depth_dir, file) for file in os.listdir(depth_dir) if file.endswith('.mat')])

    # Ensure that the number of images and depth files are equal
    assert len(image_files) == len(depth_files), "Mismatch between number of images and depth files in Make3D dataset"

    for idx in range(len(depth_files)):
        depth_path = depth_files[idx]
        depth_data = scipy.io.loadmat(depth_path)
        # Access the depth data under 'Position3DGrid' key
        depth = depth_data['Position3DGrid'][:, :, -1].astype('float32')  # Assuming depth is the last channel
        depth_list.append(depth.flatten())

    if depth_list:
        depth_data = np.concatenate(depth_list)
        return depth_data
    else:
        return np.array([])

def load_kitti_dataset(depth_dir):
    """
    Load depth data from the KITTI dataset stored in .png files.
    """
    depth_list = []
    for file in os.listdir(depth_dir):
        if file.endswith('.png'):
            depth_map = cv2.imread(os.path.join(depth_dir, file), cv2.IMREAD_UNCHANGED)
            depth_list.append(depth_map.flatten())
    if depth_list:
        depth_data = np.concatenate(depth_list)
        return depth_data/256
    else:
        return np.array([])

def main():
    # Paths to your datasets
    diode_data_dir = '/home/jiwoo/Documents/ReDepth/val/outdoor/'
    make3d_image_dir = '/home/jiwoo/Documents/ReDepth/make3d/image'  # Adjust as necessary
    make3d_depth_dir = '/home/jiwoo/Documents/ReDepth/make3d/depth/Train400Depth'
    kitti_depth_dir = '/home/jiwoo/Documents/Dataset/KITTI/dataset/sequences/10/gt_depth/image_02'

    # Load depth data from each dataset
    print("Loading DIODE dataset...")
    diode_depth = load_diode_dataset(diode_data_dir)
    print("Loading Make3D dataset...")
    make3d_depth = load_make3d_dataset(make3d_image_dir, make3d_depth_dir)
    print("Loading KITTI dataset...")
    kitti_depth = load_kitti_dataset(kitti_depth_dir)

    # Remove invalid depth values (zero or negative)
    diode_depth = diode_depth[diode_depth > 0]
    make3d_depth = make3d_depth[make3d_depth > 0]
    kitti_depth = kitti_depth[kitti_depth > 0]

    # Convert depths to meters if necessary
    # For example, if depth values are in centimeters, divide by 100
    # For Make3D dataset, depths are in meters, so no conversion needed
    # For KITTI dataset, depths might be in 16-bit format with scaling, adjust accordingly

    # Compute disparity (inverse of depth)
    diode_disparity = 1.0 / diode_depth
    make3d_disparity = 1.0 / make3d_depth
    kitti_disparity = 1.0 / kitti_depth

    # Plot the disparity distributions
    plt.figure(figsize=(12, 8))
    plt.hist(diode_disparity, bins=50, density=True, alpha=0.5, label='DIODE Dataset', color='blue')
    plt.hist(make3d_disparity, bins=50, density=True, alpha=0.5, label='Make3D Dataset', color='green')
    plt.hist(kitti_disparity, bins=50, density=True, alpha=0.5, label='KITTI Dataset', color='red')
    plt.xlabel('Disparity (1/m)')
    plt.ylabel('Density')
    plt.title('Disparity Distributions of Different Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
