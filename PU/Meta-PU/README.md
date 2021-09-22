# Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud

[[tvcg paper]](https://arxiv.org/abs/2102.04317)
[[github]](https://github.com/pleaseconnectwifi/Meta-PU)

## Dataset Preparing

Put train dataset file Patches_noHole_and_collected.h5 into model/data/, you can download it from [onedrive train data](https://portland-my.sharepoint.com/:u:/g/personal/shuquanye2-c_my_cityu_edu_hk/Ec30f3ITZwdKuPzBQnTjhssBha_M2GI76_tnvoV5o1CO-g?e=LJiycf).

Unzip and put test dataset files all_testset.zip for variable scales into model/data/all_testset/, you can download it from [onedrive test data](https://portland-my.sharepoint.com/:u:/g/personal/shuquanye2-c_my_cityu_edu_hk/EUcCveufh7VMgQOLLOeqR4MBzXX6vGWbvjenT0H0nv_Ldw?e=GkyJVT).

## Environment & Installation

This codebase was tested with the following environment configurations.

- Ubuntu 18.04
- CUDA 10.0
- python v3.7.3
- torch==1.4.0+cu100
- torchvision==0.5.0+cu100

install the dependencies:

`pip install -r requirements.txt`

install the pointnet++ module:

`python setup.py build_ext --inplace` or `pip install -e .`

## Training & Testing

Train:

`python main_gan.py --phase train --dataset model/data/Patches_noHole_and_collected.h5 --log_dir model/new --batch_size 4 --model model_res_mesh_pool --max_epoch 60 --gpu 0 --replace --FWWD --learning_rate 0.0001 --num_workers_each_gpu 3`

- You can easily reproduce the results in paper with a batch size of only 8 in a single RTX 2080 Ti (11GB). The model training may costs about 9 hours. The trained model file is provided in PointCloudToolBox and you can find them in the path `model/new`. So if you only want to upsample point cloud, you can skip this step and execute the following shell script.

Test with scale R:

`python main_gan.py --phase test --dataset model/data/all_testset/${R}/input --log_dir model/new --batch_size 4 --model model_res_mesh_pool --model_path 60 --gpu 0 --test_scale ${R}`

Evaluation with scale R:

`cd evaluation_code/`

`python evaluation_cd.py --pre_path ../model/new/result/${R}input/ --gt_path ../model/data/all_testset/${R}/gt`

## Reference
```markdown
@article{Ye2021MetaPUAA,
  title={Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud},
  author={S. Ye and Dongdong Chen and Songfang Han and Ziyu Wan and Jing Liao},
  journal={IEEE transactions on visualization and computer graphics},  
  year={2021},
  volume={PP}
}
```


