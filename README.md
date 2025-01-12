# DG-CFD

使用DG求解一维的标量方程和一维Sod激波管问题。

### 一维标量方程

$$
\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0,u(x,0)=\sin(x)
$$

### 一维欧拉方程

$$
\begin{equation}
\begin{aligned}
\text{1. 质量守恒方程:} \quad & \frac{\partial \rho(x,t)}{\partial t} + \frac{\partial}{\partial x} \left( \rho(x,t) u(x,t) \right) = 0 \\
\text{2. 动量守恒方程:} \quad & \frac{\partial (\rho(x,t) u(x,t))}{\partial t} + \frac{\partial}{\partial x} \left( \rho(x,t) u(x,t)^2 + p(x,t) \right) = 0 \\
\text{3. 能量守恒方程:} \quad & \frac{\partial E(x,t)}{\partial t} + \frac{\partial}{\partial x} \left( u(x,t) \left( E(x,t) + p(x,t) \right) \right) = 0
\end{aligned}
\end{equation}
$$

**初始条件** $t=0$
$$
\begin{cases} 
(u,\rho,p)=(0,1,1)& \text{if } x < 0 \\
(u,\rho,p)=(0,0.125,0.1) & \text{if } x \geq 0 
\end{cases}
$$



