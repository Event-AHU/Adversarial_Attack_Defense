#  Adversarial Attack for RGB-Event based Visual Object Tracking

<div align="center">

<img src="https://github.com/Event-AHU/Adversarial_Attack_Defense/blob/main/attack_rgbe_tracking/firstIMGv2.jpg" width="600">

**The first adversarial attack method specifically designed for RGB-Event based visual object tracking models** 

------
</div>

> **Adversarial Attack for RGB-Event based Visual Object Tracking**, Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang, arXiv:2504.14423 [[Paper](https://arxiv.org/abs/2504.14423)] [[Code](https://github.com/Event-AHU/Adversarial_Attack_Defense)]


# :dart: Abstract 

Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios.




# :hammer: Environment


## Framework 

<p align="center">
<img src="https://github.com/Event-AHU/Adversarial_Attack_Defense/blob/main/attack_rgbe_tracking/framework.png" alt="framework" width="700"/>
</p>







In this study, we adopt the [CEUTrack](https://github.com/Event-AHU/COESOT) model as the target tracking architecture for adversarial attack analysis. To ensure reproducibility, please configure the conda environment according to the [CEUTrack ](https://github.com/Event-AHU/COESOT)official documentation and load the pre-trained weights provided.


## Attack

```python
#Attack RGB Event voxel:
You can modify the attack parameters in Attack_RGB_Event_Voxel/experiments/ceutrack/ceutrack_coesot.yaml in order to get the raw tracking results and the results after the attack. 
Later, execute the following command:
bash Attack_RGB_Event_Voxel/test.sh
```

```python
#Attack RGB Event frame:
You can modify the attack parameters in Attack_RGB_Event_Frame/experiments/ceutrack/ceutrack_coesot.yaml in order to get the raw tracking results and the results after the attack. 
Later, execute the following command:
bash Attack_RGB_Event_Frame/test.sh
```
```

# :triangular_ruler: Evaluation Toolkit

我基于 [CEUTrack](https://github.com/Event-AHU/COESOT) 修改了对应的评测脚本。

1.修改Evaluate_COESOT_benchmark_SP_RR_Only.m中的tracking_type后运行即可获取PR、SR和NPR的结果。

2.run `Evaluate_FELT_benchmark_attributes.m` for attributes analysis and figure saved in `$/res_fig/`. 

3.run `plot_radar.m` for attributes radar figrue plot.

<p align="center">
  <img width=50%" src="https://github.com/Event-AHU/Adversarial_Attack_Defense/blob/main/attack_rgbe_tracking/attribute_analysis.jpg" alt="Radar"/>
</p>


# :cupid: Acknowledgement 

[[CEUTrack](https://github.com/Event-AHU/COESOT)] 
[[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)] 
[[FE108](https://github.com/Jee-King/ICCV2021_Event_Frame_Tracking)] 
[[OSTrack](https://github.com/botaoye/OSTrack)] 

# :newspaper: Citation 

If you think this project is helpful, please feel free to leave a star ⭐️ and cite our paper:

```bibtex
@article{chen2025adversarial,
  title={Adversarial Attack for RGB-Event based Visual Object Tracking},
  author={Chen, Qiang and Wang, Xiao and Wang, Haowen and Jiang, Bo and Zhu, Lin and Zhang, Dawei and Tian, Yonghong and Tang, Jin},
  journal={arXiv preprint arXiv:2504.14423},
  year={2025}
}
```
