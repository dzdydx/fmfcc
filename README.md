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
├─feature_extraction // 用于存放特征提取代码
├─models
│  ├─SAM
|  ├─Sinc-SENet
│  └─...
├─post-processing  // 用于模型输出后处理的脚本
├─results // 用于存放各个模型的输出，以及ground_truth
├─scripts  // 调用各个模型训练、预测的脚本
└─...
```

## Workflow

### 初始化
首先，在本机上[安装git环境](https://git-scm.com/)。

将仓库克隆到本地
```
git clone git@github.com:dzdydx/fmfcc.git
```

创建一个属于你自己的分支，并切换到该分支。分支命名为`<your_name-dev>`
```
git checkout -b <your-name>-dev
```

将你的模型代码复制到`/models/<model_name>`路径下，添加所有文件，并且提交修改
提交前整理文件夹，在`.gitignore`中排除你的数据集、保存的中间结果和模型训练的中间文件。
```
git add -A
git commit -m 'init'
```

### 提交代码

代码修改完成后，先在本地进行保存、添加和提交
```
git add -A
git commit -m '<简要描述修改了哪些内容>'
```

然后，推送到远程仓库。如果按照上述流程初始化，远程仓库应该已经默认设置完毕。
```
git push origin <your-name>-dev
```

如果已经基本完成代码的修改，可以发起pull-request，申请将分支合并到`main`

或者，可以在本地合并分支到`main`，并直接提交到远程仓库
```
git checkout main
git merge dev
git push origin main
```

此时，本地分支应该与主分支完全同步，可以删除自己的分支，或者确认代码同步后继续修改。

### 修改代码

若要修改代码，首先从主分支拉取最新代码
```
git fetch
```

然后，创建自己的分支，并切换到该分支。
```
git checkout -b <your-name>-dev
```

后续同上。

## Feature Extraction

用于存放提取特征的代码。由于官方提供的服务器/home分区空间不足，提取的特征保存到`/data/audio2`中。

## Models

目前本仓库包含一份模型的代码：
1. LCNN

## Scripts

一些有用的脚本。
1. `calc_log_loss.py`：用于计算指定JSON文件的log-loss，ground_truth来自`results/ground_truth.json`，是通过清洗我们最好的结果得到的。用法详见文件说明。
2. `linear_fusion.py`：用于将多个模型的输入线性平均，输出最终提交结果。

