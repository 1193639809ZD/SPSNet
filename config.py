# @Author: error: git config user.name & please set dead value or install git
# @Date: 2024-07-20 16:28
# @LastEditTime: 2024-09-03 15:29
# @Description: 配置文件

from pathlib import Path


class config:
    log = "logs"
    checkpoints = "checkpoints"
    datasets = Path("/home/eveleaf/MyDataset")
    whu = Path("/home/eveleaf/MyDataset/whu")
    mas = Path("/home/eveleaf/MyDataset/massachusetts")
    mas_building = Path("/home/eveleaf/MyDataset/massachusetts_building")
    yqtech = Path("/home/eveleaf/MyDataset/yqdataset")
    wayback = Path("/home/eveleaf/MyDataset/wayback")
