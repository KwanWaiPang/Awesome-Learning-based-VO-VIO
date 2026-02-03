<p align="center">
  <h1 align="center">
  Awesome Learning-based Odometry
  </h1>
</p>

This repository contains a curated list of resources addressing Learning-based visual odometry (including both image and event camera), visual-inertial odometry, inertial odometry, LiDAR odometry, Semantic SLAM, NeRF SLAM, etc.

If you find some ignored papers, **feel free to [*create pull requests*](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM/blob/pdf/How-to-PR.md), or [*open issues*](https://github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO/issues/new)**. 

Contributions in any form to make this list more comprehensive are welcome.

If you find this repository useful, a simple star should be the best affirmation. 😊

Feel free to share this list with others!

# Overview

- [Learning-based VO](#Learning-based-VO)
- [Learning-based VIO](#Learning-based-VIO)
- [Learning-based Inertial Odometry](#Learning-based-Inertial-Odometry)
- [Learning-based LiDAR Odometry](#Learning-based-LiDAR-Odometry)
- [Semantic SLAM](#Semantic-SLAM)
- [NeRF SLAM](#NeRF-SLAM)
- [Other Related Resource](#Other-Related-Resource)


## Learning-based VO

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2026|`arXiv`|[SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings](https://arxiv.org/pdf/2601.09665)|---|基于DPVO，学习patch-level 3D几何表征，通过几何调制注意力（geometry-modulated attention）机制，实现跨时间窗口的尺度一致性；通过显式的坐标约束增强重投影优化，将drift拉回到标准的尺度下|
|2026|`RAL`|[360DVO: Deep Visual Odometry for Monocular 360-Degree Camera](https://arxiv.org/pdf/2601.02309)| [![Github stars](https://img.shields.io/github/stars/chris1004336379/360DVO.svg)](https://github.com/chris1004336379/360DVO)|[website](https://chris1004336379.github.io/360DVO-homepage/)<br>基于深度学习的单目全向视觉里程计框架,畸变感知球面特征提取器(SphereResNet)+全向可微分束调整 (ODBA) 模块|
|2025|`arXiv`|[Vipe: Video pose engine for 3d geometric perception](https://arxiv.org/pdf/2508.10934)|[![Github stars](https://img.shields.io/github/stars/nv-tlabs/vipe.svg)](https://github.com/nv-tlabs/vipe)|[website](https://research.nvidia.com/labs/toronto-ai/vipe/)<br>英伟达开源的，从视频流中估算相机pose与深度；将BA（类似 DROID-SLAM的稠密光流约束）与learning深度估计相结合：语义分割移动物体+DROID-SLAM构建BA约束+单目深度估计网络（Metric3dv2）； RTX 5090，640*480分辨率，3-5HZ|
|2025|`arXiv`|[FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM](https://arxiv.org/pdf/2512.25008)|---|通过基础深度模型提供的指导，将光流估计/flow estimation与几何推理相结合； Hybrid Flow Network输出几何约束的correspondences，进而保证深度与pose估计的一致性； Bi-Consistent Bundle Adjustment Layer对多视角约束下的关键帧pose和深度进行优化；Reliability-Aware Refinement mechanism通过区分可靠与不可靠的区域实现flow update|
|2025|`arXiv`|[KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM](https://arxiv.org/pdf/2512.01889)|[![Github stars](https://img.shields.io/github/stars/be2rlab/km-vipe.svg)](https://github.com/be2rlab/km-vipe)|输入RGB数据，经GeoCalib估计相机内参、DINO提取视觉特征、自适应鲁棒kernel处理动态区域，再通过global BA优化位姿和深度，最后融合CLIP语言编码器实现开放词汇查询，标注“视觉-几何-语言全链路融合，实现“几何建图+语义理解+动态适应”|
|2025|`TRO`|[Continual Learning of Regions for Efficient Robot Localization on Large Maps](https://ieeexplore.ieee.org/abstract/document/11196045)|[![Github stars](https://img.shields.io/github/stars/MI-BioLab/continual-learning-regions.svg)](https://github.com/MI-BioLab/continual-learning-regions)|---|
|2025|`arXiv`|[Policies over Poses: Reinforcement Learning based Distributed Pose-Graph Optimization for Multi-Robot SLAM](https://arxiv.org/pdf/2510.22740)|[![Github stars](https://img.shields.io/github/stars/herolab-uga/policies-over-poses.svg)](https://github.com/herolab-uga/policies-over-poses)|---|
|2025|`arXiv`|[Real-Time Indoor Object SLAM with LLM-Enhanced Priors](https://arxiv.org/pdf/2509.21602)|---|---|
|2025|`TIM`|[A Robust and Accurate Stereo SLAM Based on Learned Feature Extraction and Matching](https://ieeexplore.ieee.org/abstract/document/11202432/keywords#keywords)|---|---|
|2025|`DAGM German Conference on Pattern Recognition`|[CoProU-VO: Combining Projected Uncertainty for End-to-End Unsupervised Monocular Visual Odometry](https://arxiv.org/pdf/2508.00568)|[![Github stars](https://img.shields.io/github/stars/Jchao-Xie/CoProU.svg)](https://github.com/Jchao-Xie/CoProU)|[website](https://jchao-xie.github.io/CoProU/)|
|2025|`RAL`|[Occlusion-Aware Monocular Visual Odometry for Robust Trajectory Tracking](https://ieeexplore.ieee.org/abstract/document/11123740/)|---|---|
|2025|`RAL`|[DINO-VO: A Feature-Based Visual Odometry Leveraging a Visual Foundation Model](https://arxiv.org/pdf/2507.13145)|---|---|
|2025|`IEEE International Symposium on Circuits and Systems`|[LoSeVO: Local Sequence Constraints for Deep Visual Odometry](https://ieeexplore.ieee.org/abstract/document/11043333/)|---|---|
|2025|`arXiv`|[VOCAL: Visual Odometry via ContrAstive Learning](https://arxiv.org/pdf/2507.00243)|---|---|
|2025|`RAL`|[Online Adaptive Keypoint Extraction for Visual Odometry Across Different Scenes](https://ieeexplore.ieee.org/abstract/document/11020750/)|---|Reinforcement Learning|
|2025|`ICRA`|[MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry](https://arxiv.org/pdf/2409.09479)|[![Github stars](https://img.shields.io/github/stars/MAC-VO/MAC-VO.svg)](https://github.com/MAC-VO/MAC-VO)|[website](https://mac-vo.github.io/)| 
|2025|`arXiv`|[Large-scale visual SLAM for in-the-wild videos](https://arxiv.org/pdf/2504.20496)|---|---|
|2025|`CVPR`|[Scene-agnostic Pose Regression for Visual Localization](https://arxiv.org/pdf/2503.19543)|[![Github stars](https://img.shields.io/github/stars/JunweiZheng93/SPR.svg)](https://github.com/JunweiZheng93/SPR)|[website](https://junweizheng93.github.io/publications/SPR/SPR.html)<br>Mamba-based|
|2025|`arXiv`|[Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](https://arxiv.org/pdf/2503.17358)|---|Velometer|
|2024|`CVPR`|[From variance to veracity: Unbundling and mitigating gradient variance in differentiable bundle adjustment layers](https://openaccess.thecvf.com/content/CVPR2024/papers/Gurumurthy_From_Variance_to_Veracity_Unbundling_and_Mitigating_Gradient_Variance_in_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/swami1995/V2V.svg)](https://github.com/swami1995/V2V)|---| 
|2024|`Complex & Intelligent Systems`|[Lp-slam: language-perceptive RGB-D SLAM framework exploiting large language model](https://link.springer.com/content/pdf/10.1007/s40747-024-01408-0.pdf)|[![Github stars](https://img.shields.io/github/stars/GroupOfLPSLAM/LP_SLAM.svg)](https://github.com/GroupOfLPSLAM/LP_SLAM)|---|
|2024|`TIV`|[Dh-ptam: a deep hybrid stereo events-frames parallel tracking and mapping system](https://arxiv.org/pdf/2306.01891)|[![Github stars](https://img.shields.io/github/stars/AbanobSoliman/DH-PTAM.svg)](https://github.com/AbanobSoliman/DH-PTAM)|Superpoint+[stereo ptam](https://github.com/uoip/stereo_ptam)|
|2024|`RAL`|[Salient sparse visual odometry with pose-only supervision](https://arxiv.org/pdf/2404.04677)|---|---| 
|2024|`RAL`|[CodedVO: Coded Visual Odometry](https://arxiv.org/pdf/2407.18240)|[![Github stars](https://img.shields.io/github/stars/naitri/CodedVO.svg)](https://github.com/naitri/CodedVO)|[website](https://prg.cs.umd.edu/CodedVO)|
|2024|`CVPR`|[Leap-vo: Long-term effective any point tracking for visual odometry](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_LEAP-VO_Long-term_Effective_Any_Point_Tracking_for_Visual_Odometry_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/chiaki530/leapvo.svg)](https://github.com/chiaki530/leapvo)|[website](https://chiaki530.github.io/projects/leapvo/)|
|2024|`RAL`|[Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning](https://arxiv.org/pdf/2408.17005)|[![Github stars](https://img.shields.io/github/stars/ShuyangUni/drl_exposure_ctrl.svg)](https://github.com/ShuyangUni/drl_exposure_ctrl)|---| 
|2024|`ECCV`|[Reinforcement learning meets visual odometry](https://arxiv.org/pdf/2407.15626)|[![Github stars](https://img.shields.io/github/stars/uzh-rpg/rl_vo.svg)](https://github.com/uzh-rpg/rl_vo)|[Blog](https://kwanwaipang.github.io/RL-for-VO/)|
|2024|`IROS`|[Deep Visual Odometry with Events and Frames](https://arxiv.org/pdf/2309.09947)|[![Github stars](https://img.shields.io/github/stars/uzh-rpg/rampvo.svg)](https://github.com/uzh-rpg/rampvo)|RAMP-VO| 
|2024|`3DV`|[Deep event visual odometry](https://arxiv.org/pdf/2312.09800)|[![Github stars](https://img.shields.io/github/stars/tum-vision/DEVO.svg)](https://github.com/tum-vision/DEVO)|---|
|2024|`CVPR`|[Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization](https://arxiv.org/pdf/2404.15263)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/MultiSlam_DiffPose.svg)](https://github.com/princeton-vl/MultiSlam_DiffPose)|Multi VO|
|2024|`ECCV`|[Deep patch visual slam](https://arxiv.org/pdf/2408.01654)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/DPVO.svg)](https://github.com/princeton-vl/DPVO)|---|
|2024|`NIPS`|[Deep patch visual odometry](https://proceedings.neurips.cc/paper_files/paper/2023/file/7ac484b0f1a1719ad5be9aa8c8455fbb-Paper-Conference.pdf)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/DPVO.svg)](https://github.com/princeton-vl/DPVO)|---|
|2023|`CVPR`|[Pvo: Panoptic visual odometry](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_PVO_Panoptic_Visual_Odometry_CVPR_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/zju3dv/pvo.svg)](https://github.com/zju3dv/pvo)|[website](https://zju3dv.github.io/pvo/)|
|2023|`ICRA`|[Dytanvo: Joint refinement of visual odometry and motion segmentation in dynamic environments](https://arxiv.org/pdf/2209.08430)|[![Github stars](https://img.shields.io/github/stars/castacks/DytanVO.svg)](https://github.com/castacks/DytanVO)|---|
|2022|`IEEE/SICE International Symposium on System Integration`|[Maskvo: Self-supervised visual odometry with a learnable dynamic mask](https://ieeexplore.ieee.org/abstract/document/9708796/)|---|---|
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
|2022|`RAL`|[SimVODIS++: Neural semantic visual odometry in dynamic environments](https://ieeexplore.ieee.org/abstract/document/9712359/)|---|---| 
|2020|`TPAMI`|[Simvodis: Simultaneous visual odometry, object detection, and instance segmentation](https://arxiv.org/pdf/1911.05939)|[![Github stars](https://img.shields.io/github/stars/Uehwan/SimVODIS.svg)](https://github.com/Uehwan/SimVODIS)|---| 
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
* Survey for Awesome Learning-based VO and VIO [Blog](https://kwanwaipang.github.io/Learning-based-VO-VIO/)
<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`TRO`|[OKVIS2-X: Open Keyframe-based Visual-Inertial SLAM Configurable with Dense Depth or LiDAR, and GNSS](https://arxiv.org/pdf/2510.04612)|[![Github stars](https://img.shields.io/github/stars/ethz-mrl/OKVIS2-X.svg)](https://github.com/ethz-mrl/OKVIS2-X)|---|
|2025|`arXiv`|[SuperPoint-SLAM3: Augmenting ORB-SLAM3 with Deep Features, Adaptive NMS, and Learning-Based Loop Closure](https://arxiv.org/pdf/2506.13089)|[![Github stars](https://img.shields.io/github/stars/shahram95/SuperPointSLAM3.svg)](https://github.com/shahram95/SuperPointSLAM3)|---|
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
|2019|`IEEE Global Communications Conference`|[LightVO: Lightweight inertial-assisted monocular visual odometry with dense neural networks](https://mingkunyang.github.io/media/LightVO.pdf)|---|---|
|2019|`IROS`|[Deepvio: Self-supervised deep learning of monocular visual inertial odometry using 3d geometric constraints](https://arxiv.org/pdf/1906.11435)|---|---| 
|2017|`AAAI`|[Vinet: Visual-inertial odometry as a sequence-to-sequence learning problem](https://arxiv.org/pdf/1701.08376)|[![Github stars](https://img.shields.io/github/stars/HTLife/VINet.svg)](https://github.com/HTLife/VINet)|non-official implementation|



## Learning-based Inertial Odometry

* Survey for Deep IMU-Bias Inference [Blog](https://kwanwaipang.github.io/Deep-IMU-Bias/)

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
|2020|`RAL`|[Tlio: Tight learned inertial odometry](https://arxiv.org/pdf/2007.01867)|[![Github stars](https://img.shields.io/github/stars/CathIAS/TLIO.svg)](https://github.com/CathIAS/TLIO)|[website](https://cathias.github.io/TLIO/)| 
|2020|`TIV`|[AI-IMU dead-reckoning](https://arxiv.org/pdf/1904.06064)|[![Github stars](https://img.shields.io/github/stars/mbrossar/ai-imu-dr.svg)](https://github.com/mbrossar/ai-imu-dr)|---| 

## Learning-based LiDAR Odometry

* Survey for Awesome Learning-based LiDAR Odometry [Blog](https://kwanwaipang.github.io/Awesome-Learning-based-LiDAR-Odometry/)
* LiDAR-based 3DGS [Paper List](https://github.com/KwanWaiPang/Awesome-3DGS-SLAM/#LiDAR-based-3DGS)

 <!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[MA-SLAM: Active SLAM in Large-Scale Unknown Environment using Map Aware Deep Reinforcement Learning](https://arxiv.org/pdf/2511.14330)|---|---|
|2025|`IEEE Sensor Journal`|[A Generative Hierarchical Optimization Framework for LiDAR Odometry Using Conditional Diffusion Models](https://ieeexplore.ieee.org/abstract/document/11026785/)|---|---|
|2025|`CVPR`|[DiffLO: Semantic-Aware LiDAR Odometry with Diffusion-Based Refinement](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DiffLO_Semantic-Aware_LiDAR_Odometry_with_Diffusion-Based_Refinement_CVPR_2025_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/HyTree7/DiffLO.svg)](https://github.com/HyTree7/DiffLO)|---| 
|2025|`arXiv`|[LIR-LIVO: A Lightweight, Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features](https://arxiv.org/pdf/2502.08676)|[![Github stars](https://img.shields.io/github/stars/IF-A-CAT/LIR-LIVO.svg)](https://github.com/IF-A-CAT/LIR-LIVO)|Fast-LIVO+AirSLAM|
|2024|`arXiv`|[LiDAR Inertial Odometry And Mapping Using Learned Registration-Relevant Features](https://arxiv.org/pdf/2410.02961)|[![Github stars](https://img.shields.io/github/stars/neu-autonomy/FeatureLIOM.svg)](https://github.com/neu-autonomy/FeatureLIOM)|learning-based select points|
|2024|`IEEE 100th Vehicular Technology Conference`|[LiDAR-OdomNet: LiDAR Odometry Network Using Feature Fusion Based on Attention](https://www.researchgate.net/profile/Parvez-Alam-20/publication/387812406_LiDAR-OdomNet_LiDAR_Odometry_Network_Using_Feature_Fusion_Based_on_Attention/links/677e43b8fb021f2a47e1e77e/LiDAR-OdomNet-LiDAR-Odometry-Network-Using-Feature-Fusion-Based-on-Attention.pdf)| [![Github stars](https://img.shields.io/github/stars/ParvezAlam123/LiDAR-OdomNet.svg)](https://github.com/ParvezAlam123/LiDAR-OdomNet)|---| 
|2024|`IROS`|[LiDAR-Visual-Inertial Tightly-coupled Odometry with Adaptive Learnable Fusion Weights](https://comrob.fel.cvut.cz/papers/iros24loc.pdf)|---|F-LOAM+VINS-Mono|
|2024|`TRO`|[PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency](https://arxiv.org/pdf/2401.09101v1)|[![Github stars](https://img.shields.io/github/stars/PRBonn/PIN_SLAM.svg)](https://github.com/PRBonn/PIN_SLAM)|---| 
|2024|`AAAI`|[DeepPointMap: Advancing LiDAR SLAM with Unified Neural Descriptors](https://arxiv.org/abs/2312.02684)|[![Github stars](https://img.shields.io/github/stars/ZhangXiaze/DeepPointMap.svg)](https://github.com/ZhangXiaze/DeepPointMap)|---| 
|2023|`ICCV`|[NeRF-LOAM: Neural Implicit Representation for Large-Scale Incremental LiDAR Odometry and Mapping](https://arxiv.org/pdf/2303.10709)|[![Github stars](https://img.shields.io/github/stars/JunyuanDeng/NeRF-LOAM.svg)](https://github.com/JunyuanDeng/NeRF-LOAM)|---| 
|2023|`ICCV`|[DELO: Deep Evidential LiDAR Odometry using Partial Optimal Transport](https://openaccess.thecvf.com/content/ICCV2023W/UnCV/papers/Ali_DELO_Deep_Evidential_LiDAR_Odometry_Using_Partial_Optimal_Transport_ICCVW_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/saali14/DELO.svg)](https://github.com/saali14/DELO)|[website](https://skazizali.com/delo.github.io/)|
|2023|`AAAI`|[Translo: A window-based masked point transformer framework for large-scale lidar odometry](https://ojs.aaai.org/index.php/AAAI/article/download/25256/25028)|[![Github stars](https://img.shields.io/github/stars/IRMVLab/TransLO.svg)](https://github.com/IRMVLab/TransLO)|---| 
|2023|`RAL`|[LONER: LiDAR Only Neural Representations for Real-Time SLAM](https://arxiv.org/abs/2309.04937)|[![Github stars](https://img.shields.io/github/stars/umautobots/LONER.svg)](https://github.com/umautobots/LONER)|---| 
|2023|`TIV`|[HPPLO-Net: Unsupervised LiDAR Odometry Using a Hierarchical Point-to-Plane Solver](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160144&tag=1)|[![Github stars](https://img.shields.io/github/stars/IMRL/HPPLO-Net.svg)](https://github.com/IMRL/HPPLO-Net)|---| 
|2022|`TPAMI`|[Efficient 3D Deep LiDAR Odometry](https://arxiv.org/abs/2111.02135)|[![Github stars](https://img.shields.io/github/stars/IRMVLab/EfficientLO-Net.svg)](https://github.com/IRMVLab/EfficientLO-Net)|---| 
|2021|`ICRA`|[Self-supervised learning of lidar odometry for robotic applications](https://arxiv.org/pdf/2011.05418)|[![Github stars](https://img.shields.io/github/stars/leggedrobotics/DeLORA.svg)](https://github.com/leggedrobotics/DeLORA)|---| 
|2021|`CVPR`|[PWCLO-Net: Deep LiDAR Odometry in 3D Point Clouds Using Hierarchical Embedding Mask Optimization](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_PWCLO-Net_Deep_LiDAR_Odometry_in_3D_Point_Clouds_Using_Hierarchical_CVPR_2021_paper.html)|[![Github stars](https://img.shields.io/github/stars/IRMVLab/PWCLONet.svg)](https://github.com/IRMVLab/PWCLONet)|---| 
|2021|`ISPRS`|[Deeplio: deep lidar inertial sensor fusion for odometry estimation](https://isprs-annals.copernicus.org/articles/V-1-2021/47/2021/isprs-annals-V-1-2021-47-2021.pdf)|[![Github stars](https://img.shields.io/github/stars/ArashJavan/DeepLIO.svg)](https://github.com/ArashJavan/DeepLIO)|---| 
|2020|`ACM international conference on multimedia`|[Lodonet: A deep neural network with 2d keypoint matching for 3d lidar odometry estimation](https://arxiv.org/pdf/2009.00164)|---|---|
|2020|`ICRA`|[Unsupervised geometry-aware deep lidar odometry](https://gisbi-kim.github.io/publications/ycho-2020-icra.pdf)|---|[website](https://sites.google.com/view/deeplo)|
|2020|`IROS`|[DMLO: Deep Matching LiDAR Odometry](https://arxiv.org/pdf/2004.03796)|---|---|
|2019|`IROS`|[Deeppco: End-to-end point cloud odometry through deep parallel neural network](https://arxiv.org/pdf/1910.11088)|---|---|
|2019|`CVPR`|[Lo-net: Deep real-time lidar odometry](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_LO-Net_Deep_Real-Time_Lidar_Odometry_CVPR_2019_paper.pdf)|---|---|
|2019|`CVPR`|[L3-net: Towards learning based lidar localization for autonomous driving](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_L3-Net_Towards_Learning_Based_LiDAR_Localization_for_Autonomous_Driving_CVPR_2019_paper.pdf)|---|---|
|2018|`IEEE International Conference on Autonomous Robot Systems and Competitions`|[CNN for IMU assisted odometry estimation using velodyne LiDAR](https://arxiv.org/pdf/1712.06352)|---|---|
|2016|`RSS workshop`|[Deep learning for laser based odometry estimation](https://nicolaia.github.io/papers/rss_16_workshop.pdf)|---|---|

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

## Semantic SLAM

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`TRO`|[RAZER: Robust Accelerated Zero-Shot 3D Open-Vocabulary Panoptic Reconstruction with Spatio-Temporal Aggregation](https://arxiv.org/pdf/2505.15373)|---|[website](https://razer-3d.github.io/)<br>构建开放词汇语义地图,FC-CLIP+ConvNeXt,支持560类物体+1306个文本类别|


## NeRF SLAM
* Survey for NeRF-based SLAM：[Blog](https://kwanwaipang.github.io/Awesome-NeRF-SLAM/)

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`TPAMI`|[SNI-SLAM++: Tightly-Coupled Semantic Neural Implicit SLAM](https://ieeexplore.ieee.org/abstract/document/11260914/)|---|---|
|2025|`CVPR`|[MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots](https://openaccess.thecvf.com//content/CVPR2025/papers/Deng_MNE-SLAM_Multi-Agent_Neural_SLAM_for_Mobile_Robots_CVPR_2025_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/dtc111111/MNESLAM.svg)](https://github.com/dtc111111/MNESLAM)|---|
|2025|`arXiv`|[MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation](https://arxiv.org/pdf/2506.18678)|[![Github stars](https://img.shields.io/github/stars/https://github.com/dtc111111/mcnslam.svg)](https://github.com/dtc111111/mcnslam)|[website](https://dtc111111.github.io/DES-dataset/)|
|2025|`arXiv`|[LRSLAM: Low-rank Representation of Signed Distance Fields in Dense Visual SLAM System](https://arxiv.org/pdf/2506.10567)|---|---|
|2025|`arXiv`|[Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](https://arxiv.org/pdf/2504.19819)|[![Github stars](https://img.shields.io/github/stars/HoangChuongNguyen/cope-nerf.svg)](https://github.com/HoangChuongNguyen/cope-nerf)|---|
|2024|`RAL`|[N3-Mapping: Normal Guided Neural Non-Projective Signed Distance Fields for Large-scale 3D Mapping](https://arxiv.org/abs/2401.03412)|[![Github stars](https://img.shields.io/github/stars/tiev-tongji/N3-Mapping.svg)](https://github.com/tiev-tongji/N3-Mapping)|---| 
|2024|`TRO`|[PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency](https://arxiv.org/pdf/2401.09101v1)|[![Github stars](https://img.shields.io/github/stars/PRBonn/PIN_SLAM.svg)](https://github.com/PRBonn/PIN_SLAM)|---| 
|2024|`AAAI`|[DeepPointMap: Advancing LiDAR SLAM with Unified Neural Descriptors](https://arxiv.org/abs/2312.02684)|[![Github stars](https://img.shields.io/github/stars/ZhangXiaze/DeepPointMap.svg)](https://github.com/ZhangXiaze/DeepPointMap)|---| 
|2024|`ICRA`|[Towards Large-Scale Incremental Dense Mapping using Robot-centric Implicit Neural Representation](https://arxiv.org/pdf/2306.10472)|[![Github stars](https://img.shields.io/github/stars/HITSZ-NRSL/RIM.svg)](https://github.com/HITSZ-NRSL/RIM)|---| 
|2024|`CVPR`|[3D LiDAR Mapping in Dynamic Environments using a 4D Implicit Neural Representation](https://www.ipb.uni-bonn.de/pdfs/zhong2024cvpr.pdf)|[![Github stars](https://img.shields.io/github/stars/PRBonn/4dNDF.svg)](https://github.com/PRBonn/4dNDF)|---| 
|2023|`ICRA`|[SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/zhong2023icra.pdf)|[![Github stars](https://img.shields.io/github/stars/PRBonn/SHINE_mapping.svg)](https://github.com/PRBonn/SHINE_mapping)|---| 
|2023|`RAL`|[LONER: LiDAR Only Neural Representations for Real-Time SLAM](https://arxiv.org/abs/2309.04937)|[![Github stars](https://img.shields.io/github/stars/umautobots/LONER.svg)](https://github.com/umautobots/LONER)|---| 
|2023|`ICCV`|[NeRF-LOAM: Neural Implicit Representation for Large-Scale Incremental LiDAR Odometry and Mapping](https://arxiv.org/pdf/2303.10709)|[![Github stars](https://img.shields.io/github/stars/JunyuanDeng/NeRF-LOAM.svg)](https://github.com/JunyuanDeng/NeRF-LOAM)|---| 



## Other Related Resource
<!-- * Survey for SLAM in Legged Robot：[Paper List](https://github.com/KwanWaiPang/Awesome-Legged-Robot-Localization-and-Mapping) -->
* Survey for Transformer-based SLAM：[Paper List](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM) 
* Survey for Diffusion-based SLAM：[Paper List](https://github.com/KwanWaiPang/Awesome-Diffusion-based-SLAM) 
* Survey for 3DGS-based SLAM: [Paper List](https://github.com/KwanWaiPang/Awesome-3DGS-SLAM)
* Survey for Dynamic SLAM: [Blog](https://kwanwaipang.github.io/Dynamic-SLAM/)
* Deep learning for image matching lecture [slides](https://cmp.felk.cvut.cz/~mishkdmy/slides/MPV2025_Learned_matching.pdf)
<!-- * Paper Survey for Degeneracy of LiDAR-SLAM [Blog](https://kwanwaipang.github.io/Lidar_Degeneracy/) -->
<!-- * Paper Survey for dynamic SLAM [Blog](https://kwanwaipang.github.io/Dynamic-SLAM/) -->
<!--  * Reproduction and Learning for LOAM Series [Blog](https://blog.csdn.net/gwplovekimi/article/details/119711762?spm=1001.2014.3001.5502) -->
* Some related papers for learning-based VO, VIO, such as datasets, depth estimation, correspondence learning, etc.

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[SupeRANSAC: One RANSAC to Rule Them All](https://arxiv.org/pdf/2506.04803)|[![Github stars](https://img.shields.io/github/stars/danini/superansac.svg)](https://github.com/danini/superansac)|---|
|2025|`ICRA`|[LiftFeat: 3D Geometry-Aware Local Feature Matching](https://arxiv.org/pdf/2505.03422)|[![Github stars](https://img.shields.io/github/stars/lyp-deeplearning/LiftFeat.svg)](https://github.com/lyp-deeplearning/LiftFeat)|---|
|2025|`arXiv`|[EdgePoint2: Compact Descriptors for Superior Efficiency and Accuracy](https://arxiv.org/pdf/2504.17280)|[![Github stars](https://img.shields.io/github/stars/HITCSC/EdgePoint2.svg)](https://github.com/HITCSC/EdgePoint2)|---| 
|2025|`arXiv`|[To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition](https://arxiv.org/pdf/2504.06116)|---|Matching|
|2025|`CVPR`|[Learning Affine Correspondences by Integrating Geometric Constraints](https://arxiv.org/pdf/2504.04834)|[![Github stars](https://img.shields.io/github/stars/stilcrad/DenseAffine.svg)](https://github.com/stilcrad/DenseAffine)|Matching/Correspondences| 
|2025|`CVPR`|[MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](https://demuc.de/papers/pataki2025mpsfm.pdf)|[![Github stars](https://img.shields.io/github/stars/cvg/mpsfm.svg)](https://github.com/cvg/mpsfm)|---|
|2025|`arXiv`|[Selecting and Pruning: A Differentiable Causal Sequentialized State-Space Model for Two-View Correspondence Learning](https://arxiv.org/pdf/2503.17938)|---|Mamba two-view correspondence|
|2024|`CVPR`|[DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector](https://openaccess.thecvf.com/content/CVPR2024W/IMW/papers/Edstedt_DeDoDe_v2_Analyzing_and_Improving_the_DeDoDe_Keypoint_Detector_CVPRW_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/Parskatt/DeDoDe.svg)](https://github.com/Parskatt/DeDoDe)|[DeDoDe v2 Pytorch Inference](https://github.com/ibaiGorordo/dedodev2-pytorch-inference)|
|2024|`3DV`|[DeDoDe: Detect, don't describe—Describe, don't detect for local feature matching](https://arxiv.org/pdf/2308.08479)|[![Github stars](https://img.shields.io/github/stars/Parskatt/DeDoDe.svg)](https://github.com/Parskatt/DeDoDe)|detectors and descriptors jointly|
|2024|`ECCV`|[Stereoglue: Robust estimation with single-point solvers](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07485.pdf)|[![Github stars](https://img.shields.io/github/stars/danini/stereoglue.svg)](https://github.com/danini/stereoglue)|stereo superglue|
|2024|`TPAMI`|[Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation](https://arxiv.org/pdf/2404.15506)|[![Github stars](https://img.shields.io/github/stars/YvanYin/Metric3D.svg)](https://github.com/YvanYin/Metric3D)|[website](https://jugghm.github.io/Metric3Dv2/)<br>depth estimation|
|2022|`NIPS`|[Theseus: A library for differentiable nonlinear optimization](https://proceedings.neurips.cc/paper_files/paper/2022/file/185969291540b3cd86e70c51e8af5d08-Paper-Conference.pdf)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/theseus.svg)](https://github.com/facebookresearch/theseus)|[website](https://sites.google.com/view/theseus-ai/)|
|2020|`CVPR`|[Superglue: Learning feature matching with graph neural networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/magicleap/SuperGluePretrainedNetwork.svg)](https://github.com/magicleap/SuperGluePretrainedNetwork)|---|
|2020|`IROS`|[Tartanair: A dataset to push the limits of visual slam](https://arxiv.org/pdf/2003.14338)|[![Github stars](https://img.shields.io/github/stars/castacks/tartanair_tools.svg)](https://github.com/castacks/tartanair_tools)|[website](https://theairlab.org/tartanair-dataset/)|
|2021|`3DV`|[RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/pdf/2109.07547)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT-Stereo.svg)](https://github.com/princeton-vl/RAFT-Stereo)|---|
|2020|`ECCV`|[Raft: Recurrent all-pairs field transforms for optical flow](https://arxiv.org/pdf/2003.12039)|[![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT.svg)](https://github.com/princeton-vl/RAFT)|---|
|2019|`CVPR`|[Projective manifold gradient layer for deep rotation regression](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Projective_Manifold_Gradient_Layer_for_Deep_Rotation_Regression_CVPR_2022_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/JYChen18/RPMG.svg)](https://github.com/JYChen18/RPMG)|Better learn rotations|
|2018|`ECCV`|[Mvsnet: Depth inference for unstructured multi-view stereo](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf)|---|---|
|2018|`CVPR`|[Superpoint: Self-supervised interest point detection and description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/rpautrat/SuperPoint.svg)](https://github.com/rpautrat/SuperPoint)|[pytorch version](https://github.com/eric-yyjau/pytorch-superpoint)<br>[event superpoint](https://github.com/mingyip/pytorch-superpoint)|
|2016|`CVPR`|[Structure-from-motion revisited](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/colmap/colmap.svg)](https://github.com/colmap/colmap)|COLMAP|
|2015|`ICCV`|[Flownet: Learning optical flow with convolutional networks](https://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf)|---|---|


* Some related papers for learning-based LiDAR odometry, such as point cloud registration, etc.

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->
| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`ICCV`|[TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](https://arxiv.org/pdf/2507.01439)|[![Github stars](https://img.shields.io/github/stars/Laka-3DV/TurboReg.svg)](https://github.com/Laka-3DV/TurboReg)|---|
|2024|`CVPR`|[Point transformer v3: Simpler faster stronger](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/Pointcept/PointTransformerV3.svg)](https://github.com/Pointcept/PointTransformerV3)|---|
|2024|`NeurIPS`|[A Consistency-Aware Spot-Guided Transformer for Versatile and Hierarchical Point Cloud Registration](https://arxiv.org/pdf/2410.10295)|[![Github stars](https://img.shields.io/github/stars/RenlangHuang/CAST.svg)](https://github.com/RenlangHuang/CAST)|---|
|2020|`CVPR`|[3dregnet: A deep neural network for 3d point registration](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pais_3DRegNet_A_Deep_Neural_Network_for_3D_Point_Registration_CVPR_2020_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/3DVisionISR/3DRegNet.svg)](https://github.com/3DVisionISR/3DRegNet)|---|
|2020|`CVPR`|[P2b: Point-to-box network for 3d object tracking in point clouds](http://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_P2B_Point-to-Box_Network_for_3D_Object_Tracking_in_Point_Clouds_CVPR_2020_paper.pdf)|---|---|
|2019|`CVPR`|[Pointconv: Deep convolutional networks on 3d point clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf)|---|---|
|2019|`ICCV`|[Meteornet: Deep learning on dynamic 3d point cloud sequences](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_MeteorNet_Deep_Learning_on_Dynamic_3D_Point_Cloud_Sequences_ICCV_2019_paper.pdf)|---|---|
|2019|`ICCV`|[Deep closest point: Learning representations for point cloud registration](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)|---|---|
|2019|`CVPR`|[Pointnetlk: Robust & efficient point cloud registration using pointnet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/hmgoforth/PointNetLK.svg)](https://github.com/hmgoforth/PointNetLK)|---|
|2019|`CVPR`|[3D local features for direct pairwise registration](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)|---|---|
|2019|`ICCV`|[Kpconv: Flexible and deformable convolution for point clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf)| [![Github stars](https://img.shields.io/github/stars/HuguesTHOMAS/KPConv.svg)](https://github.com/HuguesTHOMAS/KPConv)|---|
|2019|`ICCV`|[Deepvcp: An end-to-end deep neural network for point cloud registration](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lu_DeepVCP_An_End-to-End_Deep_Neural_Network_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)|---|---|
|2019|`CVPR`|[Flownet3d: Learning scene flow in 3d point clouds](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_FlowNet3D_Learning_Scene_Flow_in_3D_Point_Clouds_CVPR_2019_paper.pdf)|---|---|
|2019|`IROS`|[Rangenet++: Fast and accurate lidar semantic segmentation](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf)|---|---|
|2019|`ACM Transactions on Graphics`|[Dynamic graph cnn for learning on point clouds](https://dl.acm.org/doi/pdf/10.1145/3326362)|---|---|
|2018|`ECCV`|[3dfeat-net: Weakly supervised local 3d features for point cloud registration](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/yewzijian/3DFeatNet.svg)](https://github.com/yewzijian/3DFeatNet)|---|
|2017|`NIPS`|[Pointnet++: Deep hierarchical feature learning on point sets in a metric space](https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)|---|---|
|2017|`CVPR`|[Pointnet: Deep learning on point sets for 3d classification and segmentation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)|---|---|
|2015|`IROS`|[Voxnet: A 3d convolutional neural network for real-time object recognition](http://graphics.stanford.edu/courses/cs233-21-spring/ReferencedPapers/voxnet_07353481.pdf)|---|---|

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->



