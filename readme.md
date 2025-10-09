在卷积神经网络（CNN）中，卷积操作后输出特征图（feature map）的空间尺寸（高度和宽度）可以通过以下公式计算：

### 通用公式（适用于高度或宽度）：

$$
\text{Output size} = \left\lfloor \frac{\text{Input size} + 2 \times \text{Padding} - \text{Kernel size}}{\text{Stride}} \right\rfloor + 1
$$

其中：

- **Input size**：输入特征图的高度或宽度（假设为正方形或分别计算 H 和 W）
- **Padding**：在输入边界填充的像素数（通常上下左右对称，所以总填充为 $$2 \times \text{Padding}$$
- **Kernel size**：卷积核（滤波器）的大小（通常为奇数，如 3、5 等）
- **Stride**：卷积步长（每次卷积核移动的像素数）
- **⌊ ⌋**：向下取整（在大多数框架中，若不能整除则舍弃边界）

> 注意：如果使用的是“same”填充（即保持输出尺寸与输入相同），那么框架会自动计算所需的 padding。

---

### 示例：

假设输入尺寸为 \( H = 32 \)，卷积核大小 \( K = 3 \)，步长 \( S = 1 \)，填充 \( P = 1 \)：

$$
\text{Output} = \frac{32 + 2 \times 1 - 3}{1} + 1 = \frac{31}{1} + 1 = 32
$$

输出尺寸仍为 32（即“same”填充效果）。

再比如：输入 28，核大小 5，步长 2，无填充（P=0）：

$$
\text{Output} = \frac{28 + 0 - 5}{2} + 1 = \frac{23}{2} + 1 = 11.5 + 1 \rightarrow \lfloor 11.5 \rfloor + 1 = 11 + 1 = 12
$$

---

### 特殊情况说明：

- **Valid padding**：即 \( P = 0 \)，不填充。
- **Same padding**：选择 \( P \) 使得输出尺寸等于 $$\lceil \frac{\text{Input size}}{\text{Stride}} \rceil$$。对于 stride=1，通常 $$ P = \frac{K - 1}{2} $$（当 K 为奇数时）。

