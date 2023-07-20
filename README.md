# 数据集准准备
## WHU building dataset
数据集地址：http://gpcv.whu.edu.cn/data/building_dataset.html

```python
# 将下载到的数据集放到data文件夹下，命名为WHU_Building_Dataset
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