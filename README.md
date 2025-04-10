<p align="center">
  <h1 align="center">
  Awesome Learning-based VO, VIO, and IO
  </h1>
</p>

This repository contains a curated list of resources addressing Learning-based visual odometry, visual-inertial odometry, and inertial odometry

If you find some ignored papers, **feel free to [*create pull requests*](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM/blob/pdf/How-to-PR.md), or [*open issues*](https://github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO/issues/new)**. 

Contributions in any form to make this list more comprehensive are welcome.

If you find this repositorie is useful, a simple star should be the best affirmation. 😊

Feel free to share this list with others!

# Overview

- [Learning-based VO](#Learning-based-VO)
- [Learning-based VIO](#Learning-based-VIO)
- [Learning-based IO](#Learning-based-IO)
- [Learning-based LIO](#Learning-based-LIO)

## Learning-based VO

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`CVPR`|[Scene-agnostic Pose Regression for Visual Localization](https://arxiv.org/pdf/2503.19543)|[![Github stars](https://img.shields.io/github/stars/JunweiZheng93/SPR.svg)](https://github.com/JunweiZheng93/SPR)|[website](https://junweizheng93.github.io/publications/SPR/SPR.html)<br>Mamba-based|
|2025|`arXiv`|[Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](https://arxiv.org/pdf/2503.17358)|---|Velometer|
|2024|`CVPR`|[Leap-vo: Long-term effective any point tracking for visual odometry](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_LEAP-VO_Long-term_Effective_Any_Point_Tracking_for_Visual_Odometry_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/chiaki530/leapvo.svg)](https://github.com/chiaki530/leapvo)|[website](https://chiaki530.github.io/projects/leapvo/)|
|2024|`RAL`|[Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning](https://arxiv.org/pdf/2408.17005)|[![Github stars](https://img.shields.io/github/stars/ShuyangUni/drl_exposure_ctrl.svg)](https://github.com/ShuyangUni/drl_exposure_ctrl)|---| 
|2024|`ECCV`|[Reinforcement learning meets visual odometry](https://arxiv.org/pdf/2407.15626)|[![Github stars](https://img.shields.io/github/stars/uzh-rpg/rl_vo.svg)](https://github.com/uzh-rpg/rl_vo)|[Blog](https://kwanwaipang.github.io/RL-for-VO/)|
|2024|`IROS`|[Deep Visual Odometry with Events and Frames](https://arxiv.org/pdf/2309.09947)|[![Github stars](https://img.shields.io/github/stars/uzh-rpg/rampvo.svg)](https://github.com/uzh-rpg/rampvo)|RAMP-VO| 
|2024|`3DV`|[Deep event visual odometry](https://arxiv.org/pdf/2312.09800)|[![Github stars](https://img.shields.io/github/stars/tum-vision/DEVO.svg)](https://github.com/tum-vision/DEVO)|---|
|2024|`CVPR`|[Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization](https://arxiv.org/pdf/2404.15263)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/MultiSlam_DiffPose.svg)](https://github.com/princeton-vl/MultiSlam_DiffPose)|Multi VO|
|2024|`ECCV`|[Deep patch visual slam](https://arxiv.org/pdf/2408.01654)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/DPVO.svg)](https://github.com/princeton-vl/DPVO)|---|
|2024|`NIPS`|[Deep patch visual odometry](https://proceedings.neurips.cc/paper_files/paper/2023/file/7ac484b0f1a1719ad5be9aa8c8455fbb-Paper-Conference.pdf)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/DPVO.svg)](https://github.com/princeton-vl/DPVO)|---|
|2023|`ICRA`|[Dytanvo: Joint refinement of visual odometry and motion segmentation in dynamic environments](https://arxiv.org/pdf/2209.08430)|[![Github stars](https://img.shields.io/github/stars/castacks/DytanVO.svg)](https://github.com/castacks/DytanVO)|---|
|2022|`Sensor`|[Raum-vo: Rotational adjusted unsupervised monocular visual odometry](https://www.mdpi.com/1424-8220/22/7/2651)|---|---|
|2022|`Neurocomputing`|[DeepAVO: Efficient pose refining with feature distilling for deep Visual Odometry](https://arxiv.org/pdf/2105.09899)|---|---| 
|2021|`CoRL`|[Tartanvo: A generalizable learning-based vo](https://proceedings.mlr.press/v155/wang21h/wang21h.pdf)|[![Github stars](https://img.shields.io/github/stars/castacks/tartanvo.svg)](https://github.com/castacks/tartanvo)|---|
|2021|`NIPS`|[DROID-SLAM: Deep Visual SLAM for Monocular,Stereo, and RGB-D Cameras](https://proceedings.neurips.cc/paper/2021/file/89fcd07f20b6785b92134bd6c1d0fa42-Paper.pdf)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/DROID-SLAM.svg)](https://github.com/princeton-vl/DROID-SLAM)|---|
|2021|`CVPR`|[Generalizing to the open world: Deep visual odometry with online adaptation](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalizing_to_the_Open_World_Deep_Visual_Odometry_With_Online_CVPR_2021_paper.pdf)|---|---|
|2021|`arXiv`|[DF-VO: What should be learnt for visual odometry?](https://arxiv.org/pdf/2103.00933)|[![Github stars](https://img.shields.io/github/stars/Huangying-Zhan/DF-VO.svg)](https://github.com/Huangying-Zhan/DF-VO)|---|
|2021|`ICRA`|[CNN-based ego-motion estimation for fast MAV maneuvers](https://arxiv.org/pdf/2101.01841)|[![Github stars](https://img.shields.io/github/stars/tudelft/PoseNet_Planar.svg)](https://github.com/tudelft/PoseNet_Planar)|---|
|2020|`ICRA`|[Visual odometry revisited: What should be learnt?](https://arxiv.org/pdf/1909.09803)|[![Github stars](https://img.shields.io/github/stars/Huangying-Zhan/DF-VO.svg)](https://github.com/Huangying-Zhan/DF-VO)|---|
|2020|`CVPR`|[Towards better generalization: Joint depth-pose learning without posenet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Towards_Better_Generalization_Joint_Depth-Pose_Learning_Without_PoseNet_CVPR_2020_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/B1ueber2y/TrianFlow.svg)](https://github.com/B1ueber2y/TrianFlow)|---|
|2020|`CVPR`|[Diffposenet: Direct differentiable camera pose estimation](https://openaccess.thecvf.com/content/CVPR2022/papers/Parameshwara_DiffPoseNet_Direct_Differentiable_Camera_Pose_Estimation_CVPR_2022_paper.pdf)|---|---| 
|2020|`CVPR`|[D3vo: Deep depth, deep pose and deep uncertainty for monocular visual odometry](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_D3VO_Deep_Depth_Deep_Pose_and_Deep_Uncertainty_for_Monocular_CVPR_2020_paper.pdf)|---|[website](https://cvg.cit.tum.de/research/vslam/d3vo)| 
|2021|`IJCV`|[Unsupervised scale-consistent depth learning from video](https://arxiv.org/pdf/2105.11610)|[![Github stars](https://img.shields.io/github/stars/JiawangBian/SC-SfMLearner-Release.svg)](https://github.com/JiawangBian/SC-SfMLearner-Release)<br>[![Github stars](https://img.shields.io/github/stars/JiawangBian/sc_depth_pl.svg)](https://github.com/JiawangBian/sc_depth_pl)|---|
|2019|`NIPS`|[Unsupervised scale-consistent depth and ego-motion learning from monocular video](https://proceedings.neurips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)|[![Github stars](https://img.shields.io/github/stars/JiawangBian/SC-SfMLearner-Release.svg)](https://github.com/JiawangBian/SC-SfMLearner-Release)<br>[![Github stars](https://img.shields.io/github/stars/JiawangBian/sc_depth_pl.svg)](https://github.com/JiawangBian/sc_depth_pl)|---|
|2019|`ICRA`|[Ganvo: Unsupervised deep monocular visual odometry and depth estimation with generative adversarial networks](https://arxiv.org/pdf/1809.05786)|---|---|
|2019|`CVPR`|[Recurrent neural network for (un-) supervised learning of monocular video visual odometry and depth](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Recurrent_Neural_Network_for_Un-Supervised_Learning_of_Monocular_Video_Visual_CVPR_2019_paper.pdf)|-|-| 
|2019|`ICRA`|[Pose graph optimization for unsupervised monocular visual odometry](https://arxiv.org/pdf/1903.06315)|---|---|
|2018|`IJRR`|[End-to-end, sequence-to-sequence probabilistic visual odometry through deep neural networks](https://journals.sagepub.com/doi/pdf/10.1177/0278364917734298)|---|---| 
|2018|`CVPR`|[Unsupervised learning of monocular depth estimation and visual odometry with deep feature reconstruction](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhan_Unsupervised_Learning_of_CVPR_2018_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/Huangying-Zhan/Depth-VO-Feat.svg)](https://github.com/Huangying-Zhan/Depth-VO-Feat)|---|
|2018|`ICRA`|[Undeepvo: Monocular visual odometry through unsupervised deep learning](https://arxiv.org/pdf/1709.06841)|-|[website](https://senwang.gitlab.io/UnDeepVO/)|
|2017|`ICRA`|[Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks](https://arxiv.org/pdf/1709.08429)|-|[website](https://senwang.gitlab.io/DeepVO/)|
|2017|`CVPR`|[Unsupervised learning of depth and ego-motion from video](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/tinghuiz/SfMLearner.svg)](https://github.com/tinghuiz/SfMLearner)|---|
|2015|`RAL`|[Exploring representation learning with cnns for frame-to-frame ego-motion estimation](https://ieeexplore.ieee.org/abstract/document/7347378/)|---|---|


## Learning-based VIO

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[SuperEvent: Cross-Modal Learning of Event-based Keypoint Detection](https://arxiv.org/pdf/2504.00139)|[![Github stars](https://img.shields.io/github/stars/smartroboticslab/SuperEvent.svg)](https://github.com/smartroboticslab/SuperEvent)|[website](https://smartroboticslab.github.io/SuperEvent/)| 
|2025|`arXiv`|[SuperEIO: Self-Supervised Event Feature Learning for Event Inertial Odometry](https://arxiv.org/pdf/2503.22963)|[![Github stars](https://img.shields.io/github/stars/arclab-hku/SuperEIO.svg)](https://github.com/arclab-hku/SuperEIO)|[website](https://arclab-hku.github.io/SuperEIO/)|
|2025|`Robotics and Autonomous Systems`|[CUAHN-VIO: Content-and-uncertainty-aware homography network for visual-inertial odometry](https://www.sciencedirect.com/science/article/pii/S0921889024002501)|[![Github stars](https://img.shields.io/github/stars/tudelft/CUAHN-VIO.svg)](https://github.com/tudelft/CUAHN-VIO)|---|
|2025|`TRO`|[Airslam: An efficient and illumination-robust point-line visual slam system](https://arxiv.org/pdf/2408.03520)| [![Github stars](https://img.shields.io/github/stars/sair-lab/AirSLAM.svg)](https://github.com/sair-lab/AirSLAM)|---|
|2024|`arXiv`|[SF-Loc: A Visual Mapping and Geo-Localization System based on Sparse Visual Structure Frames](https://arxiv.org/pdf/2412.01500)|[![Github stars](https://img.shields.io/github/stars/GREAT-WHU/SF-Loc.svg)](https://github.com/GREAT-WHU/SF-Loc)|[test](https://kwanwaipang.github.io/SF-Loc/)|
|2024|`arXiv`|[DEIO: Deep Event Inertial Odometry](https://arxiv.org/pdf/2411.03928)|[![Github stars](https://img.shields.io/github/stars/arclab-hku/DEIO.svg)](https://github.com/arclab-hku/DEIO)|[website](https://kwanwaipang.github.io/DEIO/)|
|2024|`CVPR`|[Adaptive vio: Deep visual-inertial odometry with online continual learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Pan_Adaptive_VIO_Deep_Visual-Inertial_Odometry_with_Online_Continual_Learning_CVPR_2024_paper.pdf)|---|---|
|2024|`RAL`|[DBA-Fusion: Tightly Integrating Deep Dense Visual Bundle Adjustment with Multiple Sensors for Large-Scale Localization and Mapping](https://arxiv.org/pdf/2403.13714)|[![Github stars](https://img.shields.io/github/stars/GREAT-WHU/DBA-Fusion.svg)](https://github.com/GREAT-WHU/DBA-Fusion)|---|
|2024|`ICRA`|[DVI-SLAM: A dual visual inertial SLAM network](https://arxiv.org/pdf/2309.13814)|---|---|
|2023|`ICRA`|[Bamf-slam: bundle adjusted multi-fisheye visual-inertial slam using recurrent field transforms](https://arxiv.org/pdf/2306.01173)|---|---|
|2022|`ECCV`|[Efficient deep visual and inertial odometry with adaptive visual modality selection](https://arxiv.org/pdf/2205.06187)|[![Github stars](https://img.shields.io/github/stars/mingyuyng/Visual-Selective-VIO.svg)](https://github.com/mingyuyng/Visual-Selective-VIO)|---|
|2022|`IEEE/ASME International Conference on Advanced Intelligent Mechatronics`|[A self-supervised, differentiable Kalman filter for uncertainty-aware visual-inertial odometry](https://arxiv.org/pdf/2203.07207)|---|---| 
|2022|`Neural Networks`|[SelfVIO: Self-supervised deep monocular Visual–Inertial Odometry and depth estimation](https://www.sciencedirect.com/science/article/pii/S0893608022000752)|---|---|
|2021|`International Conference on International Joint Conferences on Artificial Intelligence`|[Unsupervised monocular visual-inertial odometry network](https://www.ijcai.org/proceedings/2020/0325.pdf)|[![Github stars](https://img.shields.io/github/stars/Ironbrotherstyle/UnVIO.svg)](https://github.com/Ironbrotherstyle/UnVIO)|---|
|2021|`IEEE International Conference on Acoustics, Speech and Signal Processing `|[Atvio: Attention guided visual-inertial odometry](https://github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO/blob/pdf/file/ATVIO_Attention_Guided_Visual-Inertial_Odometry.pdf)|---|---|
|2019|`TPAMI`|[Unsupervised deep visual-inertial odometry with online error correction for RGB-D imagery](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8691513)|---|---|
|2019|`CVPR`|[Selective sensor fusion for neural visual-inertial odometry](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.pdf)|---|---|
|2019|`IROS`|[Deepvio: Self-supervised deep learning of monocular visual inertial odometry using 3d geometric constraints](https://arxiv.org/pdf/1906.11435)|---|---| 
|2017|`AAAI`|[Vinet: Visual-inertial odometry as a sequence-to-sequence learning problem](https://arxiv.org/pdf/1701.08376)|[![Github stars](https://img.shields.io/github/stars/HTLife/VINet.svg)](https://github.com/HTLife/VINet)|non-official implementation|


## Learning-based IO

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability](https://arxiv.org/pdf/2501.15659)|[![Github stars](https://img.shields.io/github/stars/Air-IO/Air-IO.svg)](https://github.com/Air-IO/Air-IO)|[website](https://air-io.github.io/)<br>[Test](https://kwanwaipang.github.io/AirIO/)|
|2024|`TIV`|[Deep learning for inertial positioning: A survey](https://arxiv.org/pdf/2303.03757)|---|---|
|2023|`arXiv`|[End-to-end deep learning framework for real-time inertial attitude estimation using 6dof imu](https://www.researchgate.net/profile/Arman-Asgharpoor-Golroudbari/publication/368469434_END-TO-END_DEEP_LEARNING_FRAMEWORK_FOR_REAL-TIME_INERTIAL_ATTITUDE_ESTIMATION_USING_6DOF_IMU/links/63ea42cebd7860764364396a/End-to-End-Deep-Learning-Framework-for-Real-Time-Inertial-Attitude-Estimation-using-6DoF-IMU.pdf)|---|---|
|2023|`arXiv`|[AirIMU: Learning uncertainty propagation for inertial odometry](https://arxiv.org/pdf/2310.04874)|[![Github stars](https://img.shields.io/github/stars/haleqiu/AirIMU.svg)](https://github.com/haleqiu/AirIMU)|[website](https://airimu.github.io/)<br>[Test](https://kwanwaipang.github.io/AirIMU/)|
|2023|`RAL`|[Learned inertial odometry for autonomous drone racing](https://arxiv.org/pdf/2210.15287)|[![Github stars](https://img.shields.io/github/stars/uzh-rpg/learned_inertial_model_odometry.svg)](https://github.com/uzh-rpg/learned_inertial_model_odometry)|---|
|2022|`ICRA`|[Improved state propagation through AI-based pre-processing and down-sampling of high-speed inertial data](https://www.aau.at/wp-content/uploads/2022/03/imu_preprocessing.pdf)|---|---|
|2022|`RAL`|[Deep IMU Bias Inference for Robust Visual-Inertial Odometry with Factor Graphs](https://arxiv.org/pdf/2211.04517)|---|---|
|2021|`ICRA`|[IMU Data Processing For Inertial Aided Navigation:A Recurrent Neural Network Based Approach](https://arxiv.org/pdf/2103.14286)|---|---|

## Learning-based LIO

or learning-based LIVO
<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[LIR-LIVO: A Lightweight, Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features](https://arxiv.org/pdf/2502.08676)|[![Github stars](https://img.shields.io/github/stars/IF-A-CAT/LIR-LIVO.svg)](https://github.com/IF-A-CAT/LIR-LIVO)|Fast-LIVO+AirSLAM| 


## Other Related Resource
* Survey for SLAM in Legged Robot：[Paper List](https://github.com/KwanWaiPang/Awesome-Legged-Robot-Localization-and-Mapping) 
* Survey for Transformer-based SLAM：[Paper List](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM) 
* Survey for Diffusion-based SLAM：[Paper List](https://github.com/KwanWaiPang/Awesome-Diffusion-based-SLAM) 
* Survey for NeRF-based SLAM：[Blog](https://blog.csdn.net/gwplovekimi/article/details/135083274)
* Survey for 3DGS-based SLAM: [Blog](https://kwanwaipang.github.io/3DGS-SLAM/)
* Survey for Deep IMU-Bias Inference [Blog](https://kwanwaipang.github.io/Deep-IMU-Bias/)
* Paper Survey for Degeneracy of LiDAR-SLAM [Blog](https://kwanwaipang.github.io/Lidar_Degeneracy/)
* Paper Survey for dynamic SLAM [Blog](https://kwanwaipang.github.io/Dynamic-SLAM/)
* Reproduction and Learning for LOAM Series [Blog](https://blog.csdn.net/gwplovekimi/article/details/119711762?spm=1001.2014.3001.5502)
* Some related papers for learning-based VO,VIO, such as dataset, depth estimation, correspondence learning, etc.

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`CVPR`|[MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](https://demuc.de/papers/pataki2025mpsfm.pdf)|[![Github stars](https://img.shields.io/github/stars/cvg/mpsfm.svg)](https://github.com/cvg/mpsfm)|---|
|2025|`arXiv`|[Selecting and Pruning: A Differentiable Causal Sequentialized State-Space Model for Two-View Correspondence Learning](https://arxiv.org/pdf/2503.17938)|---|Mamba two-view correspondence|
|2024|`TPAMI`|[Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation](https://arxiv.org/pdf/2404.15506)|[![Github stars](https://img.shields.io/github/stars/YvanYin/Metric3D.svg)](https://github.com/YvanYin/Metric3D)|[website](https://jugghm.github.io/Metric3Dv2/)<br>depth estimation|
|2022|`NIPS`|[Theseus: A library for differentiable nonlinear optimization](https://proceedings.neurips.cc/paper_files/paper/2022/file/185969291540b3cd86e70c51e8af5d08-Paper-Conference.pdf)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/theseus.svg)](https://github.com/facebookresearch/theseus)|[website](https://sites.google.com/view/theseus-ai/)|
|2020|`CVPR`|[Superglue: Learning feature matching with graph neural networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/magicleap/SuperGluePretrainedNetwork.svg)](https://github.com/magicleap/SuperGluePretrainedNetwork)|---|
|2020|`IROS`|[Tartanair: A dataset to push the limits of visual slam](https://arxiv.org/pdf/2003.14338)|[![Github stars](https://img.shields.io/github/stars/castacks/tartanair_tools.svg)](https://github.com/castacks/tartanair_tools)|[website](https://theairlab.org/tartanair-dataset/)|
|2020|`ECCV`|[Raft: Recurrent all-pairs field transforms for optical flow](https://arxiv.org/pdf/2003.12039)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT.svg)](https://github.com/princeton-vl/RAFT)|---|
|2019|`CVPR`|[Projective manifold gradient layer for deep rotation regression](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Projective_Manifold_Gradient_Layer_for_Deep_Rotation_Regression_CVPR_2022_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/JYChen18/RPMG.svg)](https://github.com/JYChen18/RPMG)|Better learn rotations|
|2018|`ECCV`|[Mvsnet: Depth inference for unstructured multi-view stereo](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf)|---|---|
|2018|`CVPR`|[Superpoint: Self-supervised interest point detection and description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/rpautrat/SuperPoint.svg)](https://github.com/rpautrat/SuperPoint)|[pytorch version](https://github.com/eric-yyjau/pytorch-superpoint)<br>[event superpoint](https://github.com/mingyip/pytorch-superpoint)|
|2016|`CVPR`|[Structure-from-motion revisited](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/colmap/colmap.svg)](https://github.com/colmap/colmap)|COLMAP|
|2015|`ICCV`|[Flownet: Learning optical flow with convolutional networks](https://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf)|---|---|




