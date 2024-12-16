# 数据集准准备
## WHU building dataset
数据集：[下载地址](http://gpcv.whu.edu.cn/data/WHU_aerial_0.2/WHU_aerial_0.2.zip)

```python
# 新建data文件夹，将下载到的数据集放到data文件夹下，命名为WHU_Building_Dataset
# 运行whu_dataset_preprocess.py文件
python tools/whu_dataset_preprocess.py
```

# 训练
1. model_name为模型名，可选项有SPSNet和UMobileNet
2. pretrain是加载预训练模型进行训练，为空则是从头开始训练
3. spc为超像素输入通道数，与论文中保持一致，默认为3
```python
python train.py
```

# 获取模型指标
```python
python inference.py
```

# 预测图像
1. img_path: 输入图像文件夹路径
2. output_path: 输出图像文件夹路径
```python
python predict.py
```

# 模型选择
1. unetv2：whu-25
2. deeplabv3plus：whu-9, whu-6/7
3. segnet：whu-20


# 实验结果

包括yq的数据集和whu以及mas数据集的结果

## mas

0.73