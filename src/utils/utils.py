import argparse
import os
import sys
import time
import warnings
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def str2bool(v: str, strict=True) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ("true", "yes", "on", "t", "y", "1"):  # 修复原代码少逗号的bug
            return True
        elif v.lower() in ("false", "no", "off", "f", "n", "0"):
            return False
    if strict:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
    else:
        return True


def to_cuda(data, device="cuda", exclude_keys: "list[str]" = None):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, (tuple, list, set)):
        data = [to_cuda(b, device) for b in data]
    elif isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []
        for k in data.keys():
            if k not in exclude_keys:
                data[k] = to_cuda(data[k], device)
    else:
        data = data
    return data


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = "w"
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass


def get_network(arch: str, isTrain=False, continue_train=False, init_gain=0.02, pretrained=True, num_classes=2):
    """
    修复：默认输出2维（二分类），兼容旧模型的1维输出
    :param arch: 模型架构（resnet50等）
    :param isTrain: 是否训练模式
    :param continue_train: 是否续跑训练
    :param init_gain: 初始化增益
    :param pretrained: 是否加载预训练权重
    :param num_classes: 输出类别数（默认2，二分类）
    :return: 修复后的模型
    """
    if "resnet" in arch:
        from networks.resnet import ResNet

        resnet = getattr(import_module("networks.resnet"), arch)
        if isTrain:
            if continue_train:
                # ✅ 修复1：续跑训练时用指定的类别数（默认2）
                model: ResNet = resnet(num_classes=num_classes)
            else:
                model: ResNet = resnet(pretrained=pretrained)
                # ✅ 修复2：新训练时输出2维（二分类）
                model.fc = nn.Linear(2048, num_classes)  
                nn.init.normal_(model.fc.weight.data, 0.0, init_gain)
        else:
            # ✅ 修复3：测试时默认输出2维，也可手动指定1维兼容旧模型
            model: ResNet = resnet(num_classes=num_classes)
        return model
    else:
        raise ValueError(f"Unsupported arch: {arch}")


def pad_img_to_square(img: np.ndarray):
    H, W = img.shape[:2]
    if H != W:
        new_size = max(H, W)
        img = np.pad(img, ((0, new_size - H), (0, new_size - W), (0, 0)), mode="constant")
        assert img.shape[0] == img.shape[1] == new_size
    return img