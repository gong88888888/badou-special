import numpy as np
import os
import copy
from keras import layers, models, optimizers
from keras.datasets import mnist


class BPConNet:
    """
    手推卷积神经网络
    """
    """
    i1 h1 o1
    i2 h2 02
    激活函数 sigmoid 1/(1+e^-x)
    """

    def __init__(self, rate, iters):
        self.lr = rate  # 学习率
        self.layers = []
        self.iter = iters
        pass

    def add_layer(self, mode):
        """"""
        self.layers.append(mode)
        pass

    def loss(self, res, label):
        res = np.exp(res)
        res = res / np.sum(res, axis=0)
        # index = np.argmax(label)
        loss = -np.log(res[label])
        return loss

    def optimizer(self):
        """
        优化器
        :return:
        """
        pass

    def fit(self, data):
        for layer in self.layers:
            data = layer.fit(data)
        return data

    def get_dp_loss(self, res, label):
        res = np.exp(res)
        res = res / np.sum(res, axis=0)
        res[label] -= 1
        return res

        #     dp_o=
        #     return dp_o
        #     pass

    def back_propagation(self, res, label):
        layer_i = len(self.layers)
        dp_o = self.get_dp_loss(res, label)
        while layer_i > 0:
            layer_i -= 1
            dp_o = self.layers[layer_i].back_propagation(self.lr, dp_o)

    def train_mode(self, train_data):
        """
        训练 模型
        :param train_data:
        :return:
        """
        data = train_data['data']
        label = train_data['label']
        for epoch in range(self.iter):
            print('--- Epoch %d ---' % (epoch + 1))
            permutation = np.random.permutation(len(data))
            train_images = data[permutation]
            train_labels = label[permutation]

            loss = 0
            num_correct = 0
            for i, (image, label) in enumerate(zip(train_images, train_labels)):
                if i > 0 and i % 100 == 99:
                    print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (
                        i + 1, loss / 100, num_correct))
                    loss = 0
                    num_correct = 0
                image = image[:, :, np.newaxis]
                res = self.fit(image)
                l = self.loss(res, label)

                acc = 1 if np.argmax(res) == label else 0

                loss += l
                num_correct += acc

                self.back_propagation(res, label)


class ConLayer:
    # 卷积层
    def __init__(self, high, width, num, stride, deep):
        """
        :param high:  卷积核高
        :param width: 卷积核宽
        :param num:   卷积核数量
        :param stride: 卷积核步长
        :param deep 输入数据的深度
        """
        self.high = high
        self.width = width
        self.deep = deep
        self.num = num
        self.stride = stride
        self.filters = None
        self.input = None
        self.b = None
        self.init_weight()

    def init_weight(self):
        """
        初始化权重
        w: o_n * i_n
        b: o_n *1
        :return:
        """
        # 所有卷积参数构成一个3维矩阵， (num, high, width, deep)
        # 参数随机初始化，high*width减小方差
        self.filters = np.random.randn(self.num, self.high, self.width, self.deep)
        self.b = np.random.randn(self.num)

    def iterate_regions(self, image, size_h, size_w, stride):
        """
        输入：image，二维矩阵
        输出：(im_region, i, j), 所有length x width 大小的矩阵区域及对应位置索引
        """
        h, w, _ = image.shape
        h_new = (h - size_h) // stride + 1
        w_new = (w - size_w) // stride + 1
        for i in range(h_new):
            for j in range(w_new):
                im_region = image[(i * stride):(i * stride + size_h), (j * stride):(j * stride + size_w)]
                yield im_region, i, j

    def fit(self, image):
        self.input = image
        h, w, deep = image.shape
        h_new = (h - self.high) // self.stride + 1
        w_new = (w - self.width) // self.stride + 1
        out = np.zeros(shape=(h_new, w_new, self.num))
        for n in range(self.num):
            for img, i, j in self.iterate_regions(image, self.high, self.width, self.stride):
                out[i][j][n] = np.sum(img * self.filters[n,:,:,:]) + self.b[n]
        return out

    def rot180(self, array):
        new_array = array.reshape(array.size)
        new_array = new_array[::-1]
        new_array = new_array.reshape(array.shape)
        return new_array

    def back_propagation(self, learning_rate, d_out):
        """
        input: (d_L_d_out,learning_rate), (损失对输出结果的导数,学习率)
        output: d_L_d_inputs， 损失对输入数据的导数
        """

        H, W, C = d_out.shape
        d_input = np.zeros(self.input.shape)
        dw = np.zeros(self.filters.shape)
        db = np.zeros(self.b.shape)

        for n in range(self.num):
            for img, i, j in self.iterate_regions(self.input, H, W, self.stride):
                dw[n][i][j] = np.sum(img[:, :, 0] * d_out[:, :, n], axis=(0, 1))

        rot_f = self.rot180(self.filters)
        pad_d_out = np.pad(d_out, [(1, 1), (1, 1), (0, 0)], 'constant')

        for img, i, j in self.iterate_regions(pad_d_out, self.high, self.width, self.stride):
            for n in range(self.num):
                d_input[i][j] += np.sum(img[:, :, n] * rot_f[n, :, :, 0], axis=(0, 1))

        for f in range(self.num):
            db[f] = np.sum(d_out[:, :, f])

        self.filters -= learning_rate * dw
        self.b -= learning_rate * db
        return d_input


class PoolingLayer:
    # 最大池化层
    def __init__(self, high, width, num, stride):
        """
        :param high:  池化核高
        :param width: 池化核宽
        :param num:   池化核数量
        :param stride: 池化核步长
        """
        self.high = high
        self.width = width
        self.num = num
        self.stride = stride
        self.filters = None
        self.o_h = None
        self.o_w = None

        self.i_h = None
        self.i_w = None
        self.i_n = None
        self.input = None

    def iterate_regions(self, image):
        """
        输入：image，二维矩阵
        输出：(im_region, i, j), 所有length x width 大小的矩阵区域及对应位置索引
        """
        # self.i_h, self.i_w, self.i_n = image.shape

        for i in range(self.o_h):
            for j in range(self.o_w):
                im_region = image[(i * self.stride):(i * self.stride + self.high),
                            (j * self.stride):(j * self.stride + self.width)]
                yield im_region, i, j

    def fit(self, data):
        self.input = copy.deepcopy(data)
        self.i_h, self.i_w, self.i_n = data.shape
        # h, w, deep = data.shape
        self.o_h = (self.i_h - self.high) // self.stride + 1
        self.o_w = (self.i_w - self.width) // self.stride + 1
        out = np.zeros(shape=(self.o_h, self.o_w, self.i_n))
        for n in range(self.num):
            for h in range(self.o_h):
                for w in range(self.o_w):
                    # for img, i, j in self.iterate_regions(data):
                    out[h, w, n] = np.max(self.input[h * self.stride:h * self.stride + self.high,
                                          w * self.stride:w * self.stride + self.width, n])
        return out

    def back_propagation(self, learning_rate, d_out):
        """
        input: (h, w, num_filters),3维矩阵
        output: (h / 2, w / 2, num_filters).
        """

        d_input = np.zeros((self.i_h, self.i_w, self.i_n))
        for n in range(self.num):
            for h in range(0, self.i_h, self.high):
                for w in range(0, self.i_w, self.width):
                    st = np.argmax(self.input[h:h + self.high, w:w + self.width, n])
                    (idx, idy) = np.unravel_index(st, (self.high, self.width))
                    d_input[h + idy, w + idx, n] = d_out[h // self.high, w // self.width, n]

        return d_input


class Line:
    def __init__(self, i_n, o_n):
        """
        :param i_n: 输入数量
        :param o_n: 输出数量
        """
        self.i_n = i_n
        self.o_n = o_n
        self.data = None
        self.o = None
        self.dp_w = None
        self.dp_b = None
        self.w = None
        self.b = None
        self.init_weight()

    def init_weight(self):
        """
        初始化权重
        w: o_n * i_n
        b: o_n *1
        :return:
        """
        self.w = np.random.random((self.o_n, self.i_n))
        self.b = np.random.random((self.o_n, 1))

    def fit(self, data):
        self.data = data
        self.o = np.dot(self.w, data) + self.b
        # print(z)
        # self.o = self.sigmoid(z)
        return self.o

    def back_propagation(self, lr, dp_o):
        """
        dp_o  n*1
        :param lr :学习率
        :param dp_o:
        :return:
        """
        # dp_z = dp_o * self.o * (1 - self.o)
        self.dp_w = np.dot(dp_o, self.data.T)
        self.dp_b = dp_o
        dp_i = np.dot(dp_o.T, self.w)

        self.w -= lr * self.dp_w
        self.b -= lr * self.dp_b

        return dp_i.T

    # def update_params(self, lr):
    #     """
    #     :param lr: 学习率
    #     :return:
    #     """
    #     self.w -= lr * self.dp_w
    #     self.b -= lr * self.b


class FlattenLayer:
    def __init__(self):
        self.C = None
        self.W = None
        self.H = None

    def fit(self, inputs):
        self.H, self.W, self.C = inputs.shape
        return inputs.reshape(self.C * self.W * self.H, 1)

    def back_propagation(self, learning_rate, dy):
        return dy.reshape(self.H, self.W, self.C)


class ReluLayer:
    def __init__(self):
        self.input = None

    def fit(self, data):
        self.input = data
        ret = data.copy()
        ret[ret < 0] = 0
        return ret

    def back_propagation(self, learning_rate, d_out):
        d_input = d_out.copy()
        d_input[self.input < 0] = 0
        return d_input


class Data:
    def __init__(self, data_path):
        self.data_path = data_path  # 数据存储的位置
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_validate = None
        self.y_validate = None

    def create_data(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=self.data_path)
        self.x_train = train_images / 255 - 0.5
        self.y_train = train_labels
        self.x_test = test_images
        self.y_test = test_labels


if __name__ == '__main__':
    data = Data(os.path.join(os.getcwd(), 'mnist.npz'))
    data.create_data()
    net = BPConNet(0.1, 10000)

    con_layer = ConLayer(high=3, width=3, num=8, stride=1, deep=1)
    net.add_layer(con_layer)

    relu = ReluLayer()
    net.add_layer(relu)

    pool_layer = PoolingLayer(high=2, width=2, num=8, stride=2)
    net.add_layer(pool_layer)

    flatten_layer = FlattenLayer()
    net.add_layer(flatten_layer)

    line = Line(13 * 13 * 8, 10)
    net.add_layer(line)

    relu = ReluLayer()
    net.add_layer(relu)

    train_data = {'data': data.x_train, 'label': data.y_train}
    net.train_mode(train_data)
