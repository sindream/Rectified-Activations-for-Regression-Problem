# Rectified Activations for Regression Problems

This repository contains example code for the poster presentation **"ReDepth"** at **The 12th International Conference on Robot Intelligence Technology and Applications (RiTA 2024)**.

This research focuses on the **abnormal activation issue**, where the skewness of activations increases when estimating out-of-domain (OOD) data compared to the originally trained domain. The study demonstrates that performance can be improved by addressing this issue through a clipping method for activations.

The example codes provided include various regression problems, illustrating that the proposed **ReAct (Rectified Activations)** method enhances estimation performance in OOD scenarios.

---

## Examples Included

### 1. **Sine Wave Example**
This code trains a neural network on a sine wave and evaluates its performance when input values fall outside the trained range.

### 2. **MNIST Example**
MNIST is a well-known handwritten digit dataset.  
In this example:
- The model is trained on the original MNIST dataset.  
- OOD data is generated by applying transformations such as rotation, translation, and noise.  
- The ReAct method demonstrates superior results when testing with OOD data.

### 3. **Depth Estimation Example**
This example focuses on depth estimation using the KITTI dataset for training. The trained network is then tested on DIODE and Make3D datasets.  
- `mono_diode.py` and `mono_make3d.py`: Results without applying ReAct.  
- `mono_react_diode.py` and `mono_react_make3d.py`: Results with ReAct applied.

#### Data Acquisition:
- **DIODE Dataset**: Automatically downloaded during code execution.  
- **Make3D Dataset**: Can be downloaded using the [Make3D dataset link](http://make3d.cs.cornell.edu/data.html#make3d), which provides 400 training images and 300 aligned depth maps.