# FlashAttention Backward 推导

本文推导 FlashAttention 反向传播中常用的梯度公式，并解释为什么这些公式适合按块重算，而不需要在前向保存完整的注意力矩阵。

本文变量命名尽量贴近作业里的实现：

- `q`, `k`, `v` 分别对应查询、键、值
- `o` 对应前向输出
- `do` 对应上游传回的梯度，即 $\frac{\partial \mathcal{L}}{\partial O}$
- `l` 对应每一行的 log-sum-exp，即 $L_i = \log \sum_j e^{S_{ij}}$

## 1. 前向定义

先考虑单个 batch/head 内的一次 attention。记：

$$
S = \tau QK^\top + M
$$

其中：

- $Q \in \mathbb{R}^{N_q \times d}$
- $K \in \mathbb{R}^{N_k \times d}$
- $V \in \mathbb{R}^{N_k \times d_v}$
- $\tau = \frac{1}{\sqrt{d}}$
- $M$ 是 mask，对 causal attention 来说，非法位置可以看作加上 $-\infty$

softmax 和输出定义为：

$$
P = \operatorname{softmax}(S)
$$

$$
O = PV
$$

其中 softmax 是按行做的，也就是说：

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_t e^{S_{it}}}
$$

我们要推导：

$$
dQ = \frac{\partial \mathcal{L}}{\partial Q}, \quad
dK = \frac{\partial \mathcal{L}}{\partial K}, \quad
dV = \frac{\partial \mathcal{L}}{\partial V}
$$

已知上游梯度：

$$
dO = \frac{\partial \mathcal{L}}{\partial O}
$$

## 2. 由 $O = PV$ 推出 $dV$ 和 $dP$

因为

$$
O = PV
$$

所以按矩阵微分直接得到：

$$
dV = P^\top dO
$$

以及

$$
dP = dO V^\top
$$

这一步非常直接：

- $V$ 像普通矩阵乘法里的右矩阵
- $P$ 像左矩阵
- attention backward 的难点主要不在这里，而在 softmax 的反向

## 3. softmax 的行级反向

对第 $i$ 行，记：

$$
p = \operatorname{softmax}(s)
$$

其中 $s, p \in \mathbb{R}^{N_k}$。

softmax 的雅可比矩阵是：

$$
\frac{\partial p}{\partial s}=\operatorname{diag}(p) - pp^\top
$$

因此，对任意上游梯度 $dp$，都有：

$$
ds=\left(\operatorname{diag}(p) - pp^\top\right) dp
$$

把它按元素展开，可以得到更常见的形式：

$$
ds_j = p_j \left(dp_j - \sum_t p_t dp_t \right)
$$

于是对 attention 的整行来说：

$$
dS_{ij}=P_{ij}
\left(
dP_{ij} - \sum_t P_{it} dP_{it}
\right)
$$

定义每一行的标量

$$
\Delta_i = \sum_t P_{it} dP_{it}
$$

那么矩阵形式就是：

$$
dS = P \odot \left(dP - \Delta[:, None]\right)
$$

其中 $\odot$ 表示逐元素乘法。

## 4. 将 $\Delta_i$ 改写成只依赖 $dO$ 和 $O$

上面这个公式已经是正确的，但如果直接实现，会遇到一个问题：

- 你似乎需要完整的 $P$
- 还需要完整的 $dP$

FlashAttention 的关键就是把其中一些量改写成更容易重算的形式。

从

$$
dP = dO V^\top
$$

出发，有：

$$
dP_{ij} = dO_i \cdot V_j
$$

这里 $dO_i$ 是 $dO$ 的第 $i$ 行，$V_j$ 是 $V$ 的第 $j$ 行。

于是

$$
\Delta_i=\sum_j P_{ij} dP_{ij}=\sum_j P_{ij} (dO_i \cdot V_j)
$$

因为对固定的 $i$ 来说，$dO_i$ 与求和变量 $j$ 无关，可以提出去：

$$
\Delta_i=
dO_i \cdot \left(\sum_j P_{ij} V_j\right)
$$

而括号里的量正好就是第 $i$ 行输出 $O_i$：

$$
O_i = \sum_j P_{ij} V_j
$$

因此：

$$
\Delta_i = dO_i \cdot O_i
$$

也就是说，

$$
\Delta_i = \sum_c dO_{ic} O_{ic}
$$

这个结论非常重要，因为它说明：

- 不需要存整张 $P$
- 不需要先显式构造整张 $dP$
- 只要有前向输出 $O$ 和上游梯度 $dO$，就能得到每一行的 $\Delta$

于是 softmax backward 可以写成：

$$
dS_{ij}=P_{ij} \left(dP_{ij} - \Delta_i\right)
$$

其中

$$
\Delta_i = dO_i \cdot O_i
$$

## 5. 从 $dS$ 推出 $dQ$ 和 $dK$

注意分数矩阵是：

$$
S = \tau QK^\top + M
$$

mask $M$ 只是一个常量，不参与求导，因此：

$$
dQ = \tau dS K
$$

$$
dK = \tau dS^\top Q
$$

如果写成元素形式：

$$
dQ_{ic} = \tau \sum_j dS_{ij} K_{jc}
$$

$$
dK_{jc} = \tau \sum_i dS_{ij} Q_{ic}
$$

## 6. FlashAttention 为什么只存 $L_i = \log \sum_j e^{S_{ij}}$

标准 attention 如果把完整的 $S$ 或 $P$ 存下来，显存开销很大。

FlashAttention 的思路是：

1. 前向按块计算输出 $O$
2. 只存很少量的中间结果，例如每一行的 log-sum-exp：

$$
L_i = \log \sum_j e^{S_{ij}}
$$

3. backward 时按块重新算局部的 $S_{ij}$
4. 再由 $L_i$ 恢复局部的概率

由 softmax 定义：

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_t e^{S_{it}}}
$$

因为

$$
\sum_t e^{S_{it}} = e^{L_i}
$$

所以：

$$
P_{ij} = e^{S_{ij} - L_i}
$$

这就是 FlashAttention backward 里最常见的一步：

- 你不需要前向保存整张 $P$
- backward 时只要重新算出某个 tile 的 $S$
- 再结合该行保存下来的 $L_i$
- 就能重建这一小块的 $P$

## 7. 最终常用公式

整理起来，FlashAttention backward 的核心公式就是：

### 7.1 先算

$$
dV = P^\top dO
$$

$$
dP = dO V^\top
$$

### 7.2 再算每一行的

$$
\Delta_i = dO_i \cdot O_i
$$

### 7.3 然后得到

$$
dS = P \odot \left(dP - \Delta[:, None]\right)
$$

### 7.4 最后回传到 $Q, K$

$$
dQ = \tau dS K
$$

$$
dK = \tau dS^\top Q
$$

其中

$$
P_{ij} = e^{S_{ij} - L_i}
$$

而不是必须从前向缓存中直接读取整张 $P$。

## 8. 与作业变量名的对应关系

如果对照作业里的变量名，可以这样看：

- `q` 对应 $Q$
- `k` 对应 $K$
- `v` 对应 $V$
- `o` 对应 $O$
- `do` 对应 $dO$
- `l` 对应每一行的 $L_i$

因此常见的实现步骤是：

1. 前向保存 `q`, `k`, `v`, `o`, `l`
2. backward 先计算每一行的

$$
\Delta_i = \sum_c do_{ic} \, o_{ic}
$$

3. 然后按块重算某一小块 score：

$$
S_{ij}^{(\text{tile})} = \tau Q_i K_j^\top + M_{ij}
$$

4. 用 `l` 恢复这一块的概率：

$$
P_{ij}^{(\text{tile})} = e^{S_{ij}^{(\text{tile})} - L_i}
$$

5. 再按上面的公式累计这一块对 `dq`, `dk`, `dv` 的贡献

## 9. 因果 mask 的影响

如果是 causal attention，只需要把非法位置看作：

$$
S_{ij} = -\infty
\quad \text{when } j > i
$$

那么这些位置的 softmax 概率就是：

$$
P_{ij} = 0
$$

因此这些位置自然不会对：

- $dV$
- $dP$
- $dS$
- $dQ$
- $dK$

产生贡献。

所以 causal 与 non-causal 的 backward 公式本质相同，只是参与计算的 score 集合不同。

## 10. 实现视角的总结

FlashAttention backward 的关键不是重新发明梯度，而是把普通 attention 的梯度公式改写成如下形式：

1. 可以按块重算 $S$
2. 可以用 $L_i$ 恢复局部 $P$
3. 可以用

$$
\Delta_i = dO_i \cdot O_i
$$

避免存整张 $dP$ 或整张 $P$

因此，它本质上是：

- 数学上仍然等价于普通 attention backward
- 工程上改写成更省显存、更适合 tiled kernel 的形式
