## [Getting Started for ML Beginner](https://www.tensorflow.org/get_started/get_started_for_beginners)

### 鸢尾分类问题 (The Iris classification problem)

通过鸢尾的萼片(sepals)和花瓣(petals)的长宽来区分鸢尾的品种，只区分以下三种：

* 山鸢尾(Iris setosa)

* 维吉尼亚鸢尾(Iris virginica)

* 变色鸢尾(Iris versicolor)

![Iris setosa](https://www.tensorflow.org/images/iris_three_species.jpg)

采用[鸢尾花数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)，数据格式如下：

| 萼片长 | 萼片宽 | 花瓣长 | 花瓣宽 | 种类 |
| ------ | ------ | ------ | ------ | ---- |
| 6.4    | 2.8    | 5.6    | 2.2    | 2    |
| 5.0    | 2.3    | 3.3    | 1.0    | 1    |
| 4.9    | 2.5    | 4.5    | 1.7    | 2    |
| 4.9    | 3.1    | 1.5    | 0.1    | 0    |
| 5.7    | 3.8    | 1.7    | 0.3    | 0    |

其中，前四列为特征(features)，最后一列为标签(label)，也就是答案或者结果。每一行为一个样例(example)。

* 0 代表山鸢尾

* 1 代表变色鸢尾

* 2 代表维吉尼亚鸢尾

### 模型与训练 (models and training)

模型(model)代表特征和标签的关系。

训练为机器学习中模型逐步优化的过程(学习)。

* 监督学习(supervised machine learning)

    样例包含标签，鸢尾分类问题为监督学习问题

* 无监督学习(unsupervised machine learning)

    样例不包含标签

### 获取并运行示例程序 (Get and run the sample program)

1. [Install TensorFlow](https://www.tensorflow.org/install/)

1. ``pip install pandas``

1. ``git clone https://github.com/tensorflow/models``

1. ``cd models/samples/core/get_started/``

1. ``python premade_estimator.py``

*本文件目录下包含一份运行本例所需要的程序*

* 出现错误：

    ```
    AttributeError: module 'tensorflow' has no attribute 'keras'
    ```

    为tensorflow版本太低，keras需要1.4，[参考](https://github.com/tensorflow/tensorflow/issues/16614)
    ```python
    import tensorflow as tf
    print(tf.__version__)
    #结果为1.3.0，低于1.4
    ```

    更新tensorflow后成功运行
    ``pip3 install --upgrade tensorflow``

### TensorFlow编程栈 (The TensorFlow programming stack)

TensorFlow包含了不同层次的API，包括TensorFlow内核(kernel)；基于内核的python，c++，java和go的底层API；基于python的数据集(datasets)，矩阵(matrics)和层(layers)的中间层API;基于中间层的评估器(estimators)高层API：
![The TensorFlow Programming Environment](https://www.tensorflow.org/images/tensorflow_programming_environment.png)

### 程序分析

[premade_estimator.py](get_started/premade_estimator.py)包含了以下几个步骤：

* 导入并解析数据集

    通过[iris_data.py](get_started/iris_data.py)中的``load_data``函数，加载训练集和测试集，见函数注释

* 创建用于描述数据的特征列

    通过循环用训练集的列标题创建tensorflow的特征列，见程序注释

* 选择模型类型

    神经网络(Neural networks)有很多类型，本例使用全连接神经网络(fully connected neural network)

    ![Neural network](https://www.tensorflow.org/images/simple_dnn.svg)

    高层API通过指定评估器确定神经网络类型，评估器分为

    * 预建评估器(pre-made Estimators)

    * 自定义评估器(custom Estimators)

    本例使用tensorflow.estimator.DNNClassifier预建评估器，其他评估器可以[在此](https://www.tensorflow.org/api_docs/python/tf/estimator)查看

* 训练模型

    调用tf.estimator.DNNClassifier.train函数进行训练，输入通过构造tf.Dataset传入

* 评估模型有效性

    调用tf.estimator.DNNClassifier.evaluate函数在测试集上进行评估

* 使用训练后的模型进行预测

    调用tf.estimator.DNNClassifier.pridict函数进行预测
