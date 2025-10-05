import numpy as np
from convolution.conv import conv2d

def test_conv2d_basic():
    """测试基本的卷积操作"""
    # 创建简单的输入数组
    input_array = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    
    # 创建简单的卷积核
    kernel = np.array([[1, 0],
                      [0, 1]])
    
    # 执行卷积操作
    result = conv2d(input_array, kernel)
    
    # 预期结果
    expected = np.array([[6, 8],
                        [12, 14]])
    
    # 检查结果是否正确
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
    print("Basic convolution test passed!")

def test_conv2d_with_padding():
    """测试带填充的卷积操作"""
    # 创建简单的输入数组
    input_array = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    
    # 创建简单的卷积核
    kernel = np.array([[1, 0],
                      [0, 1]])
    
    # 执行带填充的卷积操作
    result = conv2d(input_array, kernel, padding=1)
    
    # 预期结果
    expected = np.array([[1, 3, 5, 3],
                        [5, 6, 8, 6],
                        [11, 12, 14, 8],
                        [7, 8, 10, 9]])
    
    # 检查结果是否正确
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
    print("Convolution with padding test passed!")

def test_conv2d_with_stride():
    """测试带步长的卷积操作"""
    # 创建简单的输入数组
    input_array = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]])
    
    # 创建简单的卷积核
    kernel = np.array([[1, 0],
                      [0, 1]])
    
    # 执行带步长的卷积操作
    result = conv2d(input_array, kernel, stride=2)
    
    # 预期结果
    expected = np.array([[7, 11],
                        [15, 19]])
    
    # 检查结果是否正确
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
    print("Convolution with stride test passed!")

if __name__ == "__main__":
    test_conv2d_basic()
    test_conv2d_with_padding()
    test_conv2d_with_stride()
    print("All tests passed!")