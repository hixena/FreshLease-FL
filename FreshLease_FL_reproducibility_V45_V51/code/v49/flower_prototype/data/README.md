# 数据集说明

`digits.npz`来自scikit-learn的`load_digits`数据集（UCI Optical Recognition of
Handwritten Digits）。原始数据共1797个8×8灰度手写数字样本。本项目使用
`random_state=20260912`进行分层80/20训练测试划分，并将像素从`0..16`缩放到`[0,1]`。

打包固定数据副本的目的是让Docker实验无需联网下载且在不同电脑上使用完全相同的测试集。

V31新增Fashion-MNIST的统一NPZ接口。为避免在仓库中隐式放入大型二进制文件，首次实验前
在项目根目录执行：

```powershell
python .\flower_prototype\prepare_fashion_mnist.py
```

脚本从Fashion-MNIST公开发布地址读取官方训练集和测试集，校验IDX结构，将像素缩放到
`[0,1]`并保存为`fashion_mnist.npz`。生成文件还记录四个原始压缩包的SHA-256摘要。
Docker镜像必须在该文件生成后重新构建；使用`-SkipBuild`不会把新数据加入旧镜像。

V47新增CIFAR-10。首次实验前执行：

```powershell
python .\flower_prototype\prepare_cifar10.py
```

脚本从多伦多大学官方地址下载Python版本，验证官方MD5，并保存为`cifar10.npz`。
CIFAR-10以`uint8`、channel-first展平的3072维特征存储以降低20容器内存峰值，加载节点
分区时再缩放到`[0,1]`。生成数据文件后必须重新构建Docker镜像。

三个数据文件遵循相同字段：`x_train`、`y_train`、`x_test`、`y_test`。特征为展平的二维
数组，模型输入统一为`float32`和`[0,1]`；标签必须是从0开始的连续整数。
