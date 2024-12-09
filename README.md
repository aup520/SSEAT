# [ACMMM2024] Sustainable Self-evolution Adversarial Training
Wenxuan Wang, Chenglei Wang, Huihui Qi, Menghao Ye, Xuelin Qian*, Peng Wang, Yanning Zhang
<p align="center">
  <img src="https://github.com/user-attachments/assets/6e4238a3-d4d2-49d1-b849-0209e0ae5186" alt="compare">
</p>

## Overview
When confronted with the challenge of ongoing generated new adversarial examples in complex and long-term multimedia applications, existing adversarial training methods struggle to adapt to iteratively updated attack methods. In contrast, our SSEAT model achieves sustainable defense performance improvements by continuously absorbing new adversarial knowledge.
![over](https://github.com/user-attachments/assets/7501e123-3a68-407a-829b-d6d5dad4231e)

## Environment Setups
Create and activate conda environment named ```SSEAT``` from our ```requirements.yaml```
```sh
conda env create -f requirements.yaml
conda activate SSEAT
```
## Data Preparation
Please download the CIFAR-10 and CIFAR-100 datasets from [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and place the adversarial samples generated using the adversarial attack algorithm into the ```/dataset``` folder..
```
/dataset
┣ 📂 CIFAR10
┃   ┣ 📂 data
┃   ┃   ┗ 📜 FGSM.pth
┃   ┃   ┗ 📜 PGD.pth
┃   ┃   ┗ 📜 SIM.pth
┃   ┃   ┗ 📜 DIM.pth
┃   ┃   ┗ 📜 VNIM.pth
┃
┣ 📂 CIFAR100
┃   ┣ 📂 data
┃   ┃   ┗ 📜 FGSM.pth
┃   ┃   ┗ 📜 PGD.pth
┃   ┃   ┗ 📜 SIM.pth
┃   ┃   ┗ 📜 DIM.pth
┃   ┃   ┗ 📜 VNIM.pth
```

## Inference
## RUN
You can set the size of the hyperparameters in run.sh
```
bash experiment.sh
```

## Citation
@inproceedings{wang2024sustainable,
  title={Sustainable Self-evolution Adversarial Training},
  author={Wang, Wenxuan and Wang, Chenglei and Qi, Huihui and Ye, Menghao and Qian, Xuelin and Wang, Peng and Zhang, Yanning},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9799--9808},
  year={2024}
}
