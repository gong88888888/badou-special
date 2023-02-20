import os
from keras import layers, models, optimizers
from keras.datasets import mnist
from keras.utils import to_categorical


class MyNet:
    def __init__(self, lr, epoch, batch_size=32):
        """
        :param lr:  学习率
        :param epoch: 迭代次数
        """
        self.lr = lr

        self.epochs = epoch
        self.batch_size = batch_size
        self.net = None

    def create_simple_net_v1(self):
        """
        # 创建简单神经网络
        :return:
        """
        self.net = models.Sequential()
        self.net.add(layer=layers.Dense(1024, activation='relu', input_shape=(28 * 28,)))
        self.net.add(layer=layers.Dense(512, activation='relu', ))
        self.net.add(layer=layers.Dense(10, activation='softmax'))

    def train_net(self, x_train, y_train):
        self.net.compile(loss='categorical_crossentropy',
                         optimizer=optimizers.SGD(lr=self.lr, momentum=0.9, nesterov=True), metrics=['accuracy'])
        self.net.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def evaluate_net(self, x_test, y_test):
        test_loss,test_acc = self.net.evaluate(x_test,y_test)
        print("test_loss :{}".format(test_loss))
        print("test_acc :{}".format(test_acc))

    def predict_net(self, x_test, y_test):
        y_predict = self.net.predict(x_test)


class Data:
    def __init__(self, data_path):
        self.data_path = data_path  # 数据存储的位置
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def create_data(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=self.data_path)
        self.x_train = train_images.reshape(-1, 28 * 28)
        self.y_train = train_labels
        self.x_test = test_images.reshape(-1, 28 * 28)
        self.y_test = test_labels

    def data_normalized(self):
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)


if __name__ == '__main__':
    # 生成数据
    data = Data(os.path.join(os.getcwd(), 'mnist.npz'))
    data.create_data()
    data.data_normalized()

    my_net = MyNet(epoch=10, lr=0.01)
    my_net.create_simple_net_v1()
    my_net.train_net(data.x_train, data.y_train)
    my_net.evaluate_net(data.x_test, data.y_test)
    # pass
