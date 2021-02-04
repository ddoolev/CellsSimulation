import numpy as np


class Point:

    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y


class Points:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def __getitem__(self, value):
        return Point(self.__x[value], self.__y[value])

    @property
    def x(self):
        return self.__x

    @property
    def size(self):
        return self.__x.size

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    def to_array(self):
        return np.array((self.__x, self.__y))

    def to_array_of_points(self):
        return self.to_array().transpose()


def create_random_point_array(size):
    x = np.random.rand(size)
    y = np.random.rand(size)
    return Points(x, y)
