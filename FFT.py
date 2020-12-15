import numpy as np
# https://whjkm.github.io/2018/08/22/%E5%8D%B7%E7%A7%AF%E5%92%8C%E5%BF%AB%E9%80%9F%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2%EF%BC%88FFT%EF%BC%89%E7%9A%84%E5%AE%9E%E7%8E%B0/

def DFT_slow(x):
    # Compute the discrete Fourier Transform of the 1D array x
    x = np.asarray(x, dtype=float)  # 转化为ndarray
    N = x.shape[0]  # 维度
    n = np.arange(N)  # 0~N组成一个一维向量
    k = n.reshape((N, 1))  # 转换为一个N维向量
    M = np.exp(-2j * np.pi * k * n / N)  # 离散傅里叶公式 -2j复数表示
    return np.dot(M, x)


def FFT(x):
    # A recursive implementation of the 1D Cooley-Tukey FFT
    x = np.asarray(x, dtype=float)  # 浅拷贝
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])  # 从0开始，2为间隔
        X_odd = FFT(x[1::2])  # 从1开始，2为间隔
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        '''
        使用/会出现下面的错误，改为// 向下取整
        TypeError: slice indices must be integers or None or have an __index__ method
        '''
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
