# Contrastive Pose
Pytorch implementation of ContrastPose: Category-Level Object Pose Estimation via Pose-Aware Contrastive Learning.

## Table of Contents  

- ContrastPose
  - Table of Content
  - Installation
  - Code Structure
  - Datasets
  - Training
    - The First Training Phase
    - The Second Training Phase
  - Evaluation
    

## Installation - From Pip
```shell
cd ContrastPose
pip install -r requirements.txt
```
## Code Structure
<details>
  <summary>[Click to expand]</summary>

- **ContrastPose**
  - **ContrastPose/backbone**: Some backbone networks used in the first training phase.
  - **ContrastPose/config**
    - **ContrastPose/config/common.py**: Some network and datasets settings for experiments.  
  - **ContrastPose/contrast**
    - **ContrastPose/contrast/Cont_split_rot.py**: Contrast learning codes for rotation.
    - **ContrastPose/contrast/Cont_split_trans.py**: Contrast learning codes for translation.
    - **ContrastPose/contrast/rnc_loss.py**: Contrast learning loss functions.
    - **ContrastPose/contrast/Rot_3DGC.py**: Backbone networks used for rotation.
    - **ContrastPose/contrast/utils.py**: Some utilities functions.
  - **ContrastPose/datasets**
    - **ContrastPose/datasets/data_augmentation.py**: Data augmentation functions.
    - **ContrastPose/datasets/load_data_contrastive.py**: Data loading functions for the first training phase.
    - **ContrastPose/datasets/load_data.py**： Data loading functions for the second training phase.
  - **ContrastPose/engine**
    - **ContrastPose/engine/organize_loss.py**: Loss terms for the second training phase.
    - **ContrastPose/engine/train_contrast_rot.py**: The first training phase for rotation.
    - **ContrastPose/engine/train_contrast_trans.py**: The first training phase for translation.
    - **ContrastPose/engine/train_estimator.py**: The second training phase.
  - **ContrastPose/evaluation**
    - **ContrastPose/evaluation/eval_utils_v1.py**: basic function for evaluation.
    - **ContrastPose/evaluation/evaluate.py**: evaluation codes to evaluate our model's performance.
    - **ContrastPose/evaluation/load_data_eval.py**: Data loading functions for the evaluation.
  - **ContrastPose/losses**
      - **ContrastPose/losses/fs_net_loss.py**: Loss functions from the FS-Net.
      - **ContrastPose/losses/geometry_loss.py**: Loss functions from the GPV-Pose.
      - **ContrastPose/losses/prop_loss.py**: Loss functions from the GPV-Pose.
      - **ContrastPose/losses/recon_loss.py**: Loss functions from the GPV-Pose.
  - **ContrastPose/mmcv**: MMCV packages.
  - **ContrastPose/network**
    - **ContrastPose/network/fs_net_repo**
        - **ContrastPose/network/fs_net_repo/Cross_Atten.py**: Cross attention module used in the second training phase.
        - **ContrastPose/network/fs_net_repo/FaceRecon.py**: The reconstruction codes from the HS-Pose.
        - **ContrastPose/network/fs_net_repo/gcn3d.py**: The 3DGCN codes from the HS-Pose.
        - **ContrastPose/network/fs_net_repo/PoseNet9D.py**: The pose estimation codes used in the second training phase.
        - **ContrastPose/network/fs_net_repo/PoseR.py**: The rotation head codes used in the second training phase.
        - **ContrastPose/network/fs_net_repo/PoseTs.py**: The translation and size heads codes used in the second training phase.
        - **ContrastPose/network/fs_net_repo/PoseTs.py**: The translation and size heads codes used in the second training phase.
    - **ContrastPose/network/Pose_Estimator.py**: The second training phase code.
  - **ContrastPose/tools**: Some neccessary functions for point cloud processing. 
</details>

## Dataset.
The datasets we used for training and testing are provided from the NOCS: Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation. 


## The First Training Phase.
Please note, some details are changed from the original paper for more efficient training. 

### The first training phase for rotation.
```shell
python -m engine/train_contrast_rot 
```
### The first training phase for translation.
```shell
python -m engine/train_contrast_trans  
```

Detailed configurations are in `config/config.py` and `script.sh`

## The Second Training Phase.
### The second training phase for the pose estimator.

```shell
python -m engine.train_estimator 
```
Detailed configurations are in `config/config.py` and `script.sh`

## Evaluation
```shell
python -m evaluation.evaluate 
```
Detailed configurations are in `config/config.py` and `script.sh`