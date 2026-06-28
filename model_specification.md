# 分层时间序列模型规范

## 1. 模型结构概览

这是一个**分层动态线性模型**，用于建模多个个体 $i=1,\ldots,N$ 在多个时间点 $t=1,\ldots,T$ 的观测数据。

### 核心特征：
- **个体异质性**：每个个体 $i$ 有独立的截距 $\alpha_i$ 和斜率 $\beta_i$
- **时间变化效应**：时间相关的乘法因子 $\gamma_t$ 和 $\delta_t$
- **两类随机效应**：随机游走 $u_{i,t}$ 和 AR(1) 过程 $p_{i,t}$
- **分层结构**：所有个体共享超参数 $\sigma_\alpha, \sigma_\beta, \sigma_r, \sigma_\eps, \rho, \sigma_{\xi,t}$

---

## 2. 完整数学模型

### 2.1 观测模型（Observation Model）

$$y_n \sim \mathcal{N}(\mu_n, \sigma_{n})$$

其中观测 $n$ 对应个体 $i_n$ 在时间 $t_n$，

$$\mu_n = \underbrace{\gamma_{t_n} \left(\alpha_{i_n} + \beta_{i_n} \cdot \text{offset}_n + u_{i_n, t_n}\right)}_{\text{趋势}} + \underbrace{\delta_{t_n} \cdot p_{i_n, t_n}}_{\text{扰动}}$$

$$\sigma_n = \sigma_{\xi, t_n}$$

**解释**：
- $\gamma_{t_n}$：趋势的时间乘法因子（时间相关的整体水平调整）
- $\delta_{t_n}$：扰动的时间乘法因子
- $\text{offset}_n$：个体-时间特定的时间偏移（中心化或标准化的时间变量）
- $\sigma_{\xi,t}$：时间 $t$ 的测量误差标准差（向量参数）

---

### 2.2 个体水平参数（Individual-level Parameters）

#### 截距和斜率
$$\alpha_i = \sigma_\alpha \cdot \alpha_{i,\text{raw}}, \quad \alpha_{i,\text{raw}} \sim \mathcal{N}(0, 1)$$

$$\beta_i = \sigma_\beta \cdot \beta_{i,\text{raw}}, \quad \beta_{i,\text{raw}} \sim \mathcal{N}(0, 1)$$

**结构**：
- $\alpha_i$：个体 $i$ 的基准水平（受 $\sigma_\alpha$ 缩放）
- $\beta_i$：个体 $i$ 的时间趋势（受 $\sigma_\beta$ 缩放）

---

### 2.3 时间变化随机效应（Time-varying Random Effects）

#### 随机游走过程（Random Walk for $u_{i,t}$）

$$u_{i,1} = \sigma_r \cdot u_{i,1,\text{raw}}, \quad u_{i,1,\text{raw}} \sim \mathcal{N}(0,1)$$

$$u_{i,t} = u_{i,t-1} + \sigma_r \cdot u_{i,t,\text{raw}}, \quad t=2,\ldots,T$$

$$u_{i,t,\text{raw}} \sim \mathcal{N}(0,1)$$

**特性**：
- 个体特定的随机游走（每个 $i$ 独立）
- 非平稳，方差随时间线性增长：$\text{Var}(u_{i,t}) = t \cdot \sigma_r^2$
- 捕捉个体的长期累积偏差

#### AR(1) 过程（AR(1) for $p_{i,t}$）

$$p_{i,1} = \sigma_\eps \cdot p_{i,1,\text{raw}}, \quad p_{i,1,\text{raw}} \sim \mathcal{N}(0,1)$$

$$p_{i,t} = \rho \cdot p_{i,t-1} + \sigma_\eps \cdot p_{i,t,\text{raw}}, \quad t=2,\ldots,T$$

$$p_{i,t,\text{raw}} \sim \mathcal{N}(0,1)$$

$$\rho \in [0.1, 0.9]$$

**特性**：
- 平稳 AR(1) 过程（当 $|\rho| < 1$）
- 长期方差：$\text{Var}(p_{i,t}) = \frac{\sigma_\eps^2}{1-\rho^2}$
- 捕捉个体的短期 (瞬态) 扰动

---

### 2.4 时间相关的乘法因子（Time-varying Multipliers）

#### Softplus 变换的参数化

$$\gamma_1 = 1, \quad \gamma_t = \text{softplus}(\gamma_{t-1,\log}), \quad t=2,\ldots,T$$

$$\delta_1 = 1, \quad \delta_t = \text{softplus}(\delta_{t-1,\log}), \quad t=2,\ldots,T$$

其中 $\text{softplus}(x) = \log(1 + e^x)$

**解释**：
- 初始化为 1（第一个时期的基准）
- 后续时期通过软正约束保持正数
- $\gamma_t$：控制趋势成分的时间演化（e.g., 增长率、衰减）
- $\delta_t$：控制扰动成分的时间演化（e.g., 波动性的时间变化）
- 使用 softplus 而非 exp，避免数值溢出且增长更平缓

#### 先验
$$\gamma_{\log,t} \sim \mathcal{N}(0, 0.5), \quad t=1,\ldots,T-1$$

$$\delta_{\log,t} \sim \mathcal{N}(0, 0.5), \quad t=1,\ldots,T-1$$

**作用**：严格的先验，防止 $\gamma_t, \delta_t$ 接近极值（0 或无穷大）

---

### 2.5 超参数先验（Hyperprior）

$$\sigma_\alpha \sim \text{Exponential}(2)$$

$$\sigma_\beta \sim \text{Exponential}(2)$$

$$\sigma_r \sim \text{Exponential}(3) \quad \text{（强压制，使随机游走波动较小）}$$

$$\sigma_\eps \sim \text{Exponential}(2)$$

$$\sigma_{\xi,t} \sim \text{Exponential}(1), \quad t=1,\ldots,T$$

$$\rho \sim \text{Uniform}(0.1, 0.9) \quad \text{（隐式的硬约束先验）}$$

---

## 3. 方差分解（Variance Decomposition in Generated Quantities）

在后验分布上，按时期 $t$ 分解观测方差的组成：

### 3.1 各成分方差

$$\text{Var}_{\alpha}(t) = \gamma_t^2 \sigma_\alpha^2$$
> 个体基准水平的贡献

$$\text{Var}_{\beta}(t) = \gamma_t^2 (t - \bar{t})^2 \sigma_\beta^2$$
> 个体时间趋势的贡献（随时间偏移平方增长）

$$\text{Var}_{\text{RW}}(t) = \gamma_t^2 \cdot t \cdot \sigma_r^2$$
> 随机游走累积的方差

$$\text{Var}_{\text{AR}(1)}(t) = \delta_t^2 \sigma_\eps^2 \frac{1-\rho^{2t}}{1-\rho^2}$$
> AR(1) 过程的瞬态方差（趋向稳态方差 $\frac{\sigma_\eps^2}{1-\rho^2}$）

$$\text{Var}_{\text{meas}}(t) = \delta_t^2 \sigma_{\xi,t}^2$$
> 时间特定的测量误差

### 3.2 总方差

$$\text{Var}_{\text{total}}(t) = \text{Var}_{\alpha}(t) + \text{Var}_{\beta}(t) + \text{Var}_{\text{RW}}(t) + \text{Var}_{\text{AR}(1)}(t) + \text{Var}_{\text{meas}}(t)$$

**用途**：
- 理解不同时期观测方差的主要驱动因素
- 评估个体异质性、时间效应、误差成分的相对重要性

---

## 4. 后验推断（Posterior Inference）

### 4.1 后验预测分布（Posterior Predictive Distribution）

$$\tilde{y}_n \sim \mathcal{N}(\mu_n^{(\text{post})}, \sigma_n^{(\text{post})})$$

其中 $\mu_n^{(\text{post})}$ 和 $\sigma_n^{(\text{post})}$ 使用后验样本计算。

**用途**：
- 后验预测检验 (PPC)
- 模型诊断与拟合度评估

### 4.2 对数似然（Log-likelihood）

$$\log p(y_n | \theta) = -\frac{1}{2}\log(2\pi\sigma_n^2) - \frac{(y_n - \mu_n)^2}{2\sigma_n^2}$$

**用途**：
- 期望对数预测密度 (ELPD) 计算
- 模型比较（LOO-IC 等）

---

## 5. 模型特征总结表

| 特征 | 说明 |
|------|------|
| **观测方程** | 加法分解：趋势 + 扰动 |
| **个体效应** | 固定斜率 $\alpha_i, \beta_i$（分层正态先验） |
| **随机游走** | 个体-时间特定，非平稳，捕捉长期偏差 |
| **AR(1)** | 个体-时间特定，平稳，捕捉短期扰动 |
| **时间乘子** | $\gamma_t, \delta_t$ 控制全局时间演化 |
| **测量误差** | 时间异方差 $\sigma_{\xi,t}$ |
| **参数化** | 非中心化 (non-centered) 以提高采样效率 |
| **约束** | $\rho \in [0.1, 0.9]$（AR(1) 平稳性 + 先验） |

---

## 6. 模型假设与局限

### 假设
1. 观测误差正态分布，方差为 $\sigma_{\xi,t}^2$
2. 个体参数与随机效应独立
3. AR(1) 平稳性（$|\rho| < 1$，代码中 $\rho \in [0.1,0.9]$）
4. 随机游走增量独立同分布

### 局限
- **线性在参数上**：$\mu_n$ 关于 $\alpha_i, \beta_i, u_{i,t}, p_{i,t}$ 线性
- **乘法结构固定**：$\gamma_t, \delta_t$ 的乘法因子不可学习其他形式
- **时间偏移外源性**：$\text{offset}_n$ 是已知的，不建模其不确定性

---

## 7. 计算注记

### 非中心化参数化
```
alpha_raw ~ N(0, 1)     →  alpha = sigma_alpha * alpha_raw
u_raw ~ N(0, 1)         →  u[i,t] = sum_{s=1}^t sigma_r * u_raw[i,s]
```
**好处**：减少采样相关性，加快 MCMC 混合

### Softplus 函数
$$\text{softplus}(x) = \log(1+e^x) \approx \begin{cases} x & x \ll 0 \\ x & x \approx 0 \\ \log(e^x) = x & x \gg 0 \end{cases}$$
**优势**：
- 数值稳定（避免 exp 溢出）
- 光滑且单调递增
- 界限不同时行为自然（接近恒等或 $\log e^x$）
