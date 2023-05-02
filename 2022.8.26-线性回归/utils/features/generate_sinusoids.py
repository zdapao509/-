import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    sin(x).对数据进行特征扩容，每个值都进行 sinusoid_degree次sin（），然后进行拼接
    """

    num_examples = dataset.shape[0]#确定行数，也就是样本数
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
        
    return sinusoids

# a=np.empty((5, 1))
# print(a)