#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    # 创建特征列，是一个tensorflow.feature_column.numeric_column的列表
    # pandas.DataFrame.keys()获取各列标题（轴信息）,见https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.keys.html#pandas.DataFrame.keys
    my_feature_columns = [] # python list
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    # 使用提供的评估器，包含两层隐含层，每层10个神经元，输入根据特征列，输出为三类
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    # 训练模型，调用tf.estimator.DNNClassifier.train，函数原型见https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#train
    # input_fn是输入函数，输入函数应返回特征-张量元组或者特征名-张量字典或包含张量label的张量字典
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size), # input_fn为一个匿名函数，没有参数，返回值为iris_data.train_input_fn的返回值
        steps=args.train_steps) # 训练迭代次数，默认为1000，由上面的Parser指定

    # Evaluate the model.
    # 评估模型有效性，函数原型见https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#evaluate
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    # 预测，返回predictions张量
    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    # 分别输出每个样例的预测值
    # zip函数将可迭代对象对应的元素打包成一个个元组，返回这些元组的列表
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        # class_ids为只包含一个元素的数组，元素为可能性最大的预测结果
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO) # 设置输出日志的等级，从高到低依次为FATAL, ERROR, WARN, INFO, DEBUG, 见https://www.tensorflow.org/api_docs/python/tf/logging/set_verbosity
    tf.app.run(main) # 执行main函数，https://www.tensorflow.org/api_docs/python/tf/app/run
