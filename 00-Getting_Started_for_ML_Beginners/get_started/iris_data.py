import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    # 下载训练数据集
    # tensorflow.keras是TensorFlow的高层API，实现了Keras API
    # Keras（官网 keras.io）是用Python写的高层神经网络API，兼容TensorFlow,CNTK,Theano
    # tf.kears.utils.get_file函数将指定URL的文件下载到缓存中，默认下载到~/.keras/dataset/fname（~为用户目录，对应windows下C:\Users\<用户名>\），函数原型为：
    # get_file(
    #     fname, #文件名称
    #     origin,#文件URL
    #     untar=False,
    #     md5_hash=None,
    #     file_hash=None,
    #     cache_subdir='datasets',
    #     hash_algorithm='auto',
    #     extract=False,
    #     archive_format='auto',
    #     cache_dir=None
    # )
    # 参见：https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)#split结果可运行test.py
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """
    Returns the iris dataset as (train_x, train_y), (test_x, test_y).
    """
    # 获取训练集和测试集，如果文件不存在则从TRAIN_URL和TEST_URL下载
    train_path, test_path = maybe_download()
    
    # 解析csv文件
    # pandas为python数据分析库，官网为https://pandas.pydata.org/
    # pandas.read_csv读取csv文件到DataFrame中，见http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    # 返回结果为pandas.DataFrame类型，是一种可变大小的二维表格，包含行列的名称，见https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    # 将Species列从数据集中分离
    # pandas.DataFrame.pop见https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pop.html#pandas.DataFrame.pop
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    # 创建tf.dataset数据集，https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    # 随机打乱（Shuffle）,设置重复，设置批大小
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    # 返回数据集
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    # 与train_input_fn类似
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size) # 无须打乱或者重复

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
