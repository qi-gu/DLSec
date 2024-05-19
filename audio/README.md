# AI-Testing 语音模块

![docker-ci](https://github.com/ydc123/AI-Testing/actions/workflows/docker-ci-audio.yml/badge.svg?branch=ydc)

## 下载及使用

### STEP 1. 获取项目

克隆本项目并安装依赖：

```bash
git clone https://github.com/ydc123/AI-Testing.git
cd AI-Testing/audio
pip install requirements.txt
```

或者从启智社区上克隆本项目

```bash
git clone https://openi.pcl.ac.cn/Numbda/AI-Testing.git
cd AI-Testing/audio
pip install requirements.txt
```

### STEP 2. 数据准备

本开源项目的[`audio/Datasets/`](audio/Datasets/)中提供了一小部分LibriSpeech数据集。用户可使用上述数据集，或添加其他wav格式音频数据，进行数据集扩展。

### STEP 3. 快速开始

使用`cd audio/test`进入test目录。重明开源项目提供了一个示例的算法文件test.py。可以测试在多种攻击算法下，Deep Speech模型的鲁棒性结果：

```python
python test.py
```

上述命令将测试Deep Speech模型，在FGSM、PGD、CW、遗传算法以及Imperceptible CW算法攻击下，模型预测的结果。
