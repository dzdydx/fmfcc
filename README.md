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
├─scripts  // 调用各个模型训练、预测的脚本
└─...
```

## Workflow

### 初始化
首先，在本机上[安装git环境](https://git-scm.com/)。

将仓库克隆到本地
```
git clone git@github.com:lin-lang/fmfcc.git
```

创建一个属于你自己的分支，并切换到该分支。分支命名为`你的名字缩写-dev`
```
git checkout -b <your-name>-dev
```

将你的模型代码复制到`/models/<model_name>`路径下，添加所有文件，并且提交修改
请注意控制代码库的大小，在`.gitignore`中排除你的数据集、保存的中间结果和模型训练的中间文件。
```
git add -A
git commit -m 'init'
```

### 提交代码

代码修改完成后，先在本地进行保存、添加和提交
```
git add -A
git commit -m '<简要描述你修改了哪些内容>'
```

然后，推送到远程仓库。如果你按照上述流程初始化，远程仓库应该已经默认设置完毕。
```
git push origin <your-name>-dev
```

如果已经基本完成代码的修改，可以发起pull-request，申请将分支合并到`main`

此时，你负责的那部分代码应该已经与主分支完全同步，你可以删除自己的分支。
```
git branch delete <your-name>-dev
```

### 修改代码

若要修改代码，首先从主分支拉取最新代码
```
git fetch
```

然后，创建自己的分支，并切换到该分支。分支命名为`你的名字缩写-dev`
```
git checkout -b <your-name>-dev
```

接着，你就可以在自己的分支上随意地进行修改了。如果修改完成需要合并到主分支，按照提交代码的流程操作即可。

## Models

目前本仓库包含两份模型的代码：
1. [SAM](https://www.paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1)
2. Sinc-SENet（登凯毕业论文的代码）