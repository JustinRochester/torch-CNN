# 神经网络反向传播推导

## DNN

设神经网络层数为 $L$ ，第 $l$ 层输入为 $\boldsymbol z^l=\bold W^l\cdot \boldsymbol a^{l-1}+\boldsymbol b^l$ ，输出为 $\boldsymbol a^l=f(\boldsymbol z^l)$ ，损失函数为 $C$

其中，$\boldsymbol a^0$ 为输入层，$\boldsymbol a^L$ 为输出层

---

$\boldsymbol a^L$ 的梯度有 $\displaystyle (\boldsymbol {\delta a}^L)_i={\partial C\over \partial \boldsymbol a_i^L}$ ，所以为 $\displaystyle \boldsymbol {\delta a}^L={\partial C\over \partial \boldsymbol a^L}$ 

$\boldsymbol a^l$ 的梯度有 $\displaystyle (\boldsymbol {\delta a}^l)_i=\sum_j {\partial C\over \partial \boldsymbol z^{l+1}_j}\cdot {\partial \boldsymbol z^{l+1}_j\over \partial \boldsymbol a^l_i}=\sum_j\boldsymbol {\delta z}^{l+1}_j \cdot {\partial \over \partial \boldsymbol a^l_i}\sum_k (\bold W^{l+1}_{j, k}\cdot \boldsymbol a^l_k+\boldsymbol b^{l+1}_k)=\sum_j \boldsymbol {\delta z}^{l+1}_j \bold W^{l+1}_{j, i}$ 

所以 $\displaystyle \boldsymbol{\delta a}^l=\sum_i \boldsymbol e_i\sum_j(\bold W^{l+1})^T_{i, j}\cdot \boldsymbol e_j^T\cdot \boldsymbol e_j\cdot \boldsymbol {\delta z}^{l+1}_j=(\bold W^{l+1})^T\cdot \boldsymbol {\delta z}^{l+1}$

$\boldsymbol z^l$ 的梯度有 $\displaystyle \boldsymbol {\delta z}^l_i=\sum_i\sum_j{\partial C\over \partial \boldsymbol a^l_j}\cdot {\partial \boldsymbol a^l_j\over \partial \boldsymbol z^l_i}=(\delta \boldsymbol a^l)_i\cdot f'(\boldsymbol z^l_i)$ ，所以为 $\boldsymbol \delta^l=\boldsymbol {\delta a}^l \odot f'(\boldsymbol z^l)$

$\bold W^l$ 的梯度有 $\displaystyle \bold {\delta W}^l_{i, j}=\sum_k {\partial C\over \partial \boldsymbol z^l_k}\cdot {\partial \over \partial \bold W^l_{i, j}}\sum_t(\bold W^l_{k, t}\cdot \boldsymbol a^{l-1}_t+\boldsymbol b^{l-1}_k)=\boldsymbol {\delta z}^l_i\cdot \boldsymbol a^{l-1}_j$ 
所以 $\displaystyle \bold {\delta W}^l=\sum_i\sum_j \boldsymbol e_i\cdot \boldsymbol e_j^T\cdot \boldsymbol {\delta z}^l_i\cdot \boldsymbol a^{l-1}_j=\boldsymbol {\delta z}^l\cdot (\boldsymbol a^{l-1})^T$ 

$\boldsymbol b^l$ 的梯度有 $\displaystyle (\boldsymbol {\delta b}^l)_i=\sum_k {\partial C\over \partial \boldsymbol z^l_k}\cdot {\partial \over \partial \boldsymbol b^l_i}\sum_t(\bold W^l_{k, t}\cdot \boldsymbol a^{l-1}_t+\boldsymbol b^{l-1}_k)=(\boldsymbol {\delta z}^l)_i$ ，所以为 $\boldsymbol {\delta b}^l=\boldsymbol {\delta z}^l$ 

总结反向传播梯度如下：

| forward                                                      | backword                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\boldsymbol a^L=C(\boldsymbol a^L)$                         | $\displaystyle \boldsymbol {\delta a}^L={\partial C\over \partial \boldsymbol a^L}$ |
| $\boldsymbol z^{l+1}=\bold W^{l+1}\cdot \boldsymbol a^l+\boldsymbol b^{l+1}$ | $\displaystyle \boldsymbol{\delta a}^l=(\bold W^{l+1})^T\cdot \boldsymbol {\delta z}^{l+1}$ |
| $\boldsymbol a^l=f(\boldsymbol z^l)$                         | $\boldsymbol {\delta z}^l=\boldsymbol {\delta a}^l \odot f'(\boldsymbol z^l)$ |
| $\boldsymbol z^l=\bold W^l\cdot \boldsymbol a^{l-1}+\boldsymbol b^l$ | $\bold {\delta W}^l=\boldsymbol {\delta z}^l\cdot (\boldsymbol a^{l-1})^T$ |
| $\boldsymbol z^l=\bold W^l\cdot \boldsymbol a^{l-1}+\boldsymbol b^l$ | $\boldsymbol {\delta b}^l=\boldsymbol {\delta z}^l$          |

---

## CNN 卷积层

设卷积层输入为张量 $\bold X\in \mathbb R^{N\times I\times H\times W}$ ，卷积核为张量 $\bold F\in \mathbb R^{O\times I\times FH\times FW}$ ，卷积核偏置项为 $\boldsymbol b\in \mathbb R^{O\times 1}$ ，则输出为张量 $\bold Y\in \mathbb R^{N\times O\times H'\times W'}$ 满足：

$\bold Y=\bold X\otimes \bold F+\boldsymbol b$ （含广播）

---

### 前向传播

$\bold X\in \mathbb R^{N\times I\times H\times W}\xrightarrow{\text{img2col}}\tilde {\bold X}\in \mathbb R^{(N\times H'\times W')\times (I\times FH\times FW)}$

$\bold F\in\mathbb R^{O\times I\times FH\times FW}\xrightarrow{\text{reshape}}\tilde{\bold F}\in \mathbb R^{O\times (I\times FH\times FW)}\xrightarrow{\text{T}}\tilde{\bold F}\in \mathbb R^{(I\times FH\times FW)\times O}$

$\boldsymbol b\in \mathbb R^{O\times 1}\xrightarrow{\text{T}}\tilde{\bold B}\in \mathbb R^{1\times O}\xrightarrow{\text{broadcast}}\tilde{\bold B}\in \mathbb R^{(N\times H'\times W')\times O}$

则有 $\tilde{\bold X}\cdot \tilde{\bold F}+\tilde{\bold B}=\tilde{\bold Y}\in \mathbb R^{(N\times H'\times W')\times O}$

$\tilde{\bold Y}\in \mathbb R^{(N\times H'\times W')\times O}\xrightarrow{\text{reshape}}\bold Y\in \mathbb R^{N\times H'\times W'\times O}\xrightarrow{\text{transpose}}\bold Y\in \mathbb R^{N\times O\times H'\times W'}$

---

### 反向传播

首先根据上述 DNN 的反向传播规则得到：

$\bold {\delta \tilde X}=\bold{\delta \tilde Y}\cdot (\bold {\tilde F})^T$

$\bold{\delta \tilde F}=(\bold{\tilde X})^T\cdot \bold{\delta \tilde Y}$

$\bold{\delta \tilde B}=\bold{\delta \tilde Y}$

因此反向传播规则如下：

$\bold{\delta Y}\in \mathbb R^{N\times O\times H'\times W'}\xrightarrow{\text{transpose}}\bold{\delta \tilde Y}\in \mathbb R^{N\times H'\times W'\times O}\xrightarrow{\text{reshape}}\bold{\delta \tilde Y}\in \mathbb R^{(N\times H'\times W')\times O}$

$\bold{\delta \tilde B}=\bold{\delta \tilde Y}\in \mathbb R^{(N\times H'\times W')\times O}\xrightarrow{\text{sum in axis=0}}\bold{\delta \tilde B}\in \mathbb R^{1\times O}\xrightarrow{\text{T}}\boldsymbol {\delta b}\in \mathbb R^{O\times 1}$

$\bold{\delta\tilde F}=\bold{\delta\tilde X}^T\cdot \bold{\delta\tilde Y}\in \mathbb R^{(I\times FH\times FW)\times O}\xrightarrow{\text{T}}\bold{\delta\tilde F}\in \mathbb R^{O\times (I\times FH\times FW)}\xrightarrow{\text{reshape}}\bold{\delta F}\in\mathbb R^{O\times I\times FH\times FW}$

$\bold {\delta \tilde X}=\bold{\delta \tilde Y}\cdot (\bold {\tilde F})^T\in \mathbb R^{(N\times H'\times W')\times (I\times FH\times FW)}\xrightarrow{\text{col2img}}\bold{\delta X}\in \mathbb R^{N\times I\times H\times W}$

---

## MSE

$\displaystyle L=\sqrt{\sum_i(\boldsymbol y_i-\boldsymbol o_i)^2}$

$\displaystyle {\partial L\over \partial \boldsymbol o_i}={-2(\boldsymbol y_i-\boldsymbol o_i)\over 2L}={\boldsymbol o_i-\boldsymbol y_i\over L}$

$\displaystyle {\partial L\over \partial \boldsymbol o}={1\over L}(\boldsymbol o-\boldsymbol y)$

---

## Cross Entropy-Softmax

$\displaystyle L=-\sum_i (\boldsymbol y_i\log \text{softmax}(\boldsymbol o)_i)+R$

$\displaystyle {\partial L\over \partial \text{softmax}(\boldsymbol o)_i}=-{\boldsymbol y_i\over \text{softmax}(\boldsymbol o)_i}$

$$
\begin{aligned}
    &{\partial \text{softmax}(\boldsymbol o)_i\over \partial \boldsymbol o_j}
    \\=&{\partial \over \partial \boldsymbol o_j}{\exp \boldsymbol o_i\over \sum_k \exp \boldsymbol o_k}
    \\=&{[i=j]\exp \boldsymbol o_i\sum_k \exp \boldsymbol o_k-\exp \boldsymbol o_i\cdot \exp \boldsymbol o_j\over (\sum_k \exp \boldsymbol o_k)^2}
    \\=&[i=j]\text{softmax}(\boldsymbol o)_i-\text{softmax}(\boldsymbol o)_i\cdot \text{softmax}(\boldsymbol o)_j
\end{aligned}
$$

$$
\begin{aligned}
    &{\partial L\over \partial \boldsymbol o_i}
    \\=&\sum_j {\partial L\over \partial \text{softmax}(\boldsymbol o)_j}{\partial \text{softmax}(\boldsymbol o)_j\over \partial \boldsymbol o_i}
    \\=&\sum_j-{\boldsymbol y_j\over \text{softmax}(\boldsymbol o)_j}\cdot ([i=j]\text{softmax}(\boldsymbol o)_i-\text{softmax}(\boldsymbol o)_i\cdot \text{softmax}(\boldsymbol o)_j)
    \\=&-\boldsymbol y_i+\text{softmax}(\boldsymbol o)_i\cdot \sum_j \boldsymbol y_j
\end{aligned}
$$

$\displaystyle {\partial L\over \partial \boldsymbol o}=\text{softmax}(\boldsymbol o)-\boldsymbol y$ 

---

## Batch Normalization

$\displaystyle \boldsymbol \mu={1\over m}\sum_{i=1}^m\boldsymbol x_i$

$\displaystyle \boldsymbol \sigma^2={1\over m}\sum_{i=1}^m(\boldsymbol x_i-\boldsymbol \mu)^2$

$\displaystyle \hat{\boldsymbol x}_i={\boldsymbol x_i-\boldsymbol \mu\over \sqrt{\boldsymbol \sigma^2+\epsilon}}$

$\boldsymbol y_i=\hat{\boldsymbol x}_i\odot \boldsymbol \gamma+\boldsymbol \beta$

$$
\begin{aligned}
    &{\partial C\over \partial \boldsymbol \gamma}
    \\=&\sum_i\boldsymbol e_i({\partial C\over \partial \boldsymbol \gamma})_i
    \\=&\sum_i\boldsymbol e_i\sum_j({\partial C\over \partial \boldsymbol y_j})_i({\partial \boldsymbol y_j\over \partial \boldsymbol \gamma})_i
    \\=&\sum_j{\partial C\over \partial \boldsymbol y_j}\odot\hat{\boldsymbol x}_j
\end{aligned}
$$

$$
\begin{aligned}
    &{\partial C\over \partial \boldsymbol \beta}
    \\=&\sum_i\boldsymbol e_i({\partial C\over \partial \boldsymbol \beta})_i
    \\=&\sum_i\boldsymbol e_i\sum_j({\partial C\over \partial \boldsymbol y_j})_i({\partial \boldsymbol y_j\over \partial \boldsymbol \beta})_i
    \\=&\sum_j{\partial C\over \partial \boldsymbol y_j}
\end{aligned}
$$

$$
\begin{aligned}
    &{\partial C\over \partial \hat {\boldsymbol x}_p}
    \\=&\sum_i\boldsymbol e_i({\partial C\over \partial \hat {\boldsymbol x}_p})_i
    \\=&\sum_i\boldsymbol e_i\sum_j({\partial C\over \partial \boldsymbol y_j})_i({\partial \boldsymbol y_j\over \partial \hat {\boldsymbol x}_p})_i
    \\=&{\partial C\over \partial \boldsymbol y_p}\odot\boldsymbol \gamma
\end{aligned}
$$

$$
\begin{aligned}
    &{\partial C\over \partial \boldsymbol \sigma^2}
    \\=&\sum_p\boldsymbol e_p({\partial C\over \partial \boldsymbol \sigma^2})_p
    \\=&\sum_p\boldsymbol e_p\sum_i({\partial C\over \partial \hat{\boldsymbol x}_i})_p\cdot ({\partial \hat{\boldsymbol x}_i\over \partial \sqrt{\boldsymbol \sigma^2+\epsilon}})_p\cdot ({\partial \sqrt{\boldsymbol \sigma^2+\epsilon}\over \partial \boldsymbol \sigma^2})_p
    \\=&\sum_p\boldsymbol e_p\sum_i({\partial C\over \partial \hat{\boldsymbol x}_i})_p\cdot (\boldsymbol x_i-\boldsymbol \mu)_p\cdot ({-1\over \boldsymbol \sigma^2+\epsilon})_p\cdot({1\over 2\sqrt{\boldsymbol \sigma^2+\epsilon}})_p
    \\=&-{1\over 2(\boldsymbol \sigma^2+\epsilon)}\odot \sum_i{\partial C\over \partial \hat{\boldsymbol x_i}}\odot \hat{\boldsymbol x}_i
\end{aligned}
$$

$$
\begin{aligned}
    &{\partial C\over \partial \boldsymbol \mu}
    \\=&\sum_p\boldsymbol e_p({\partial C\over \partial \boldsymbol \mu})_p
    \\=&\sum_p\boldsymbol e_p\sum_i({\partial C\over \partial \hat{\boldsymbol x}_i})_p\cdot ({\partial \hat{\boldsymbol x}_i\over \partial \boldsymbol \mu})_p+\sum_p\boldsymbol e_p({\partial C\over \partial \boldsymbol \sigma^2})_p\cdot ({\partial \boldsymbol \sigma^2\over \partial \boldsymbol \mu})_p
    \\=&-{1\over \sqrt{\boldsymbol \sigma^2+\epsilon}}\odot\sum_i{\partial C\over \partial \hat{\boldsymbol x}_i}+{\partial C\over \partial \boldsymbol \sigma^2}\odot {-2\over m}\sum_i(\boldsymbol x_i-\boldsymbol \mu)
    \\=&-{1\over \sqrt{\boldsymbol \sigma^2+\epsilon}}\odot\sum_i{\partial C\over \partial \hat{\boldsymbol x}_i}-{2\sqrt{\boldsymbol \sigma^2+\epsilon}\over m}\odot {\partial C\over \partial \boldsymbol \sigma^2}\odot \sum_i\hat{\boldsymbol x}_i
\end{aligned}
$$

$$
\begin{aligned}
&{\partial C\over \partial \boldsymbol x_i}
\\=&\sum_p \boldsymbol e_p\cdot ({\partial C\over \partial \boldsymbol x_i})_p
\\=&\sum_p\boldsymbol e_p\cdot [({\partial C\over \partial \hat{\boldsymbol x}_i})_p\cdot ({\partial \hat{\boldsymbol x}_i\over \partial \boldsymbol x_i})_p+({\partial C\over \partial \boldsymbol \mu})_p\cdot ({\partial \boldsymbol \mu\over \partial \boldsymbol x_i})_p+({\partial C\over \partial \boldsymbol \sigma^2})_p\cdot ({\partial  \boldsymbol \sigma^2\over \partial \boldsymbol x_i})_p]
\\=&{\partial C\over \partial \hat{\boldsymbol x}_i}\odot {1\over \sqrt{\boldsymbol \sigma^2+\epsilon}}+{\partial C\over \partial \boldsymbol \mu}\cdot {1\over m}+{\partial C\over \partial \boldsymbol \sigma^2}\odot {2\over m}(\boldsymbol x_i-\boldsymbol \mu)
\\=&{\partial C\over \partial \hat{\boldsymbol x}_i}\odot {1\over \sqrt{\boldsymbol \sigma^2+\epsilon}}+{\partial C\over \partial \boldsymbol \mu}\cdot {1\over m}+{2\over m}\cdot {\partial C\over \partial \boldsymbol \sigma^2}\odot \hat{\boldsymbol x}_i\odot \sqrt{\boldsymbol \sigma^2+\epsilon}
\end{aligned}
$$

