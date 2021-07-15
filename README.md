# FMFCC
用于共享数据标注结果、模型代码和后处理脚本。

整个仓库的文件结构如下所示：

```
fmfcc
├─data
│  ├─FMFCC_Audio_test
│  │  ├─test_audio
│  │  │   └─Dev
│  │  └─test_label.json  // 测试集的标注
│  │
│  └─FMFCC_Audio_train
│     ├─train_audio
│     │    └─Train
│     └─train_label.json
├─models
│  ├─SAM
|  ├─Sinc-SENet
│  └─...
├─post-processing  // 用于模型输出后处理的脚本
├─results // 用于存放各个模型的输出，以及ground_truth
├─scripts  // 调用各个模型训练、预测的脚本
└─...
```

## Models

目前本仓库包含两份模型的代码：
1. [SAM](https://www.paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1)
2. Sinc-SENet（登凯毕业论文的代码，尚未修改适配本任务）

## Scripts

一些有用的脚本。
1. `calc_log_loss.py`：用于计算指定JSON文件的log-loss，ground_truth来自`results/ground_truth.json`，是通过清洗我们最好的结果得到的。用法详见文件说明。
