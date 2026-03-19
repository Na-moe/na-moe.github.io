$$
\begin{aligned}
  A&=\tilde{\boldsymbol{\Lambda}}\!\left( XW_{Q}{W_{K}}^{\top}X^{\top} \right) \\
  o_{h} &= A(XW_{V}^{(h)}) \\
  Y &= [o_{1},\cdots,o_{h},\cdots,o_{H}]W_{o}
\end{aligned}
$$

```python
Q, K, V = X@W_Q, X@W_K, X@W_V
QK = einsum('hLd,hLd->hLL', Q, K)
M = (aL:=arange(L) <= aL[None,:])
A = softmax(QK/(d**-(1/2))) * M
O = einsum('hLL,hLd->L(h*d)', A, V)
Y = O @ W_O
```

```python
A @ X # 1 L d
AX @ W_v # L d, h d d_h -> h L d_h # bmm
O @ W_o # h L d_h, h d_h d -> h L d # bmm 
```