# Paddle-FSL-MatchingNet

## 一、简介
论文：《Matching Networks for One Shot Learning》[论文链接](https://dl.acm.org/doi/10.5555/3157382.3157504)

论文采用了基于深度神经特征的度量学习和利用外部记忆增强神经网络的技术。它学习了一个网络，将一个小的带标签的支持集和一个未带标签的示例映射到它的标签上，从而避免了调整以适应新的类类型的需要。然后定义了视觉(使用Omniglot, ImageNet)和语言任务的一次性学习问题。与竞争对手的方法相比，算法在ImageNet上的一次射击精度从87.6%提高到93.2%，在Omniglot上从88.0%提高到93.8%。

[参考项目地址链接](https://github.com/wyharveychen/CloserLookFewShot)
## 二、复现精度
代码在miniImageNet数据集下训练和测试。

5-way Acc：

| |1-shot|5-shot|
| :---: | :---: | :---: |
|论文|46.6% |60.0%|
|复现|48.3% |62.2%|

## 三、数据集
2016年google DeepMind团队从Imagnet数据集中抽取的一小部分（大小约3GB）制作了Mini-Imagenet数据集，共有100个类别，每个类别都有600张图片，共60000张（都是.jpg结尾的文件）。

Mini-Imagenet数据集中还包含了train.csv、val.csv以及test.csv三个文件。

* train.csv包含38400张图片，共64个类别。
* val.csv包含9600张图片，共16个类别。
* test.csv包含12000张图片，共20个类别。

每个csv文件之间的图像以及类别都是相互独立的，即共60000张图片，100个类。


## 四、环境依赖
paddlepaddle-gpu==2.2.2

## 五、快速开始

本项目5-way分类可设1-shot和5-shot。如果用5-shot可设置--n_shot 5，用1-shot可设置--n_shot 1。下面以5-shot为例。

### step1: 加载数据集
下载MiniImagenet数据集文件放在本repo的./filelists下

可以在这里下载[MiniImagenet数据集](https://aistudio.baidu.com/aistudio/datasetdetail/138415)


### step2: 训练

```bash
python3 train.py --n_shot 5
```

训练的模型保存在本repo的./record目录下

训练的日志保存在本repo的./logs目录下

### step3: 保存特征

将提取的特征保存在分类层之前，以提高测试速度。加载./record目录下模型进行特征保存

```bash
python3 save_features.py --n_shot 5
```

### step4: 测试

```bash
python3 test.py --n_shot 5
```

测试时程序会加载本repo的./record下保存的训练模型文件。

可以[下载训练好的模型数据](https://aistudio.baidu.com/aistudio/datasetdetail/140016)，放到本repo的./record下。

然后直接执行第step3保存特征和第step4测试命令

## 六、代码结构与参数说明

### 6.1 代码结构

```
├─data                            # 数据处理包
├─filelists                       # 数据文件
├─methods                         # 模型方法
├─logs                            # 训练日志
├─record                          # 训练保存文件    
│  configs.py                     # 配置文件
│  README.md                      # readme
│  save_features.py               # 保存特征
│  train.py                       # 训练
│  test.py                        # 测试
│  utils.py                       # 工具文件
```
### 6.2 参数说明

可以在configs.py文件中查看设置训练与测试相关参数

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | Lieber |
| 时间 | 2022.04 |
| 框架版本 | Paddle 2.2.2 |
| 应用场景 | 小样本 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [最优模型](https://aistudio.baidu.com/aistudio/datasetdetail/140016)|
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/3832241)|