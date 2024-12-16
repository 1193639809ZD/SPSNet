import torch

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值
L\P     P    N
P      TP    FN
N      FP    TN

predict 和 label 应该都是(batch_size, img_size, img_size), 里面是标签值
"""


# 添加Remove background选项，去除ignore_labels选项
class SegmentationMetric(object):
    def __init__(self, num_class, device, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = []
        self.num_class = num_class
        self.ignore_labels = ignore_labels
        # 创建空的混淆矩阵
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class)).to(device)

    def pixel_accuracy(self):
        # PA：返回分类正确的像素占总像素的比例
        # PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_precision(self):
        # return each category pixel precision
        # pre = (TP) / (TP + FP)
        class_pre = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=0)
        return class_pre  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def mean_pixel_precision(self):

        class_pre = self.class_pixel_precision()
        mean_pre = class_pre[class_pre < float('inf')].mean()  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return mean_pre  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def class_pixel_recall(self):
        # return each category pixel precision
        # pre = (TP) / TP + FP
        class_rec = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        return class_rec  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测召回率

    def mean_pixel_recall(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        class_rec = self.class_pixel_recall()
        mean_rec = class_rec[class_rec < float('inf')].mean()  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return mean_rec  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def intersection_over_union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - self.confusion_matrix.diag()
        class_iou = intersection / union  # 返回列表，其值为各个类别的IoU
        return class_iou

    def mean_intersection_over_union(self):
        iou = self.intersection_over_union()
        # 去除忽略标签的IoU
        valid_label = [label for label in range(self.num_class) if label not in self.ignore_labels]
        iou = iou[valid_label]
        miou = iou[iou < float('inf')].mean()  # 求各类别IoU的平均
        return miou

    def frequency_weighted_intersection_over_union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = self.confusion_matrix.sum(dim=1) / self.confusion_matrix.sum()
        iu = self.confusion_matrix.diag() / self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(
            dim=0) - self.confusion_matrix.diag()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fw_iou

    def f1_score(self):
        # 计算方式为Macro-F1
        # 获取pre列表和rec列表
        class_pre = self.class_pixel_precision()
        class_rec = self.class_pixel_recall()
        intersection = 2 * class_pre * class_rec
        union = class_pre + class_rec
        f1_score = intersection / union
        return f1_score

    def mean_f1_score(self):
        f1_score = self.f1_score()
        mean_f1_score = f1_score[f1_score < float('inf')].mean()
        return mean_f1_score

    def get_confusion_matrix(self, img_predict, img_label):
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param img_predict:
        :param img_label:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (img_label >= 0) & (img_label < self.num_class)
        for IgLabel in self.ignore_labels:
            mask &= (img_label != IgLabel)
        label = self.num_class * img_label[mask] + img_predict[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.view(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, img_predict, img_label):
        assert img_predict.shape == img_label.shape
        self.confusion_matrix += self.get_confusion_matrix(img_predict, img_label)  # 得到混淆矩阵
        return self.confusion_matrix

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class))
