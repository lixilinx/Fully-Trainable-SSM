# Fully Trainable SSMs

## Causal 1D SSMs 
A causal 1D SSM is defined as 

$$ x_t =  Ax_{t-1} + B u_t, \quad  y_t =  C x_t + D u_t $$

where $u$, $x$ and $y$ are the input, state and output vectors, respectively. In many works, the state matrix $A$ is fixed to certain constant ones, say the HiPPO matrix. Actually, it is the most important matrix in an SSM accounting for its long term memories. We should learn it from data. There are two possible modal decompositions suitable for highly efficient fully trainable SSM learnings. 

### Modal decomposition I 
The eigenvalue decomposition (EVD) form. With EVD $A = V\Lambda V^{-1}$, we can introduce a (generally) complex state vector $V^{-1}x$ to diagonalize the state matrix. This is the easiest way to learn $A$, and training is fast. However, there may be some waste of degrees of freedoms.

### Modal decomposition II

Another modal decomposition is to stick to the domain of real numbers strictly. Say that $A$ has a pair of complex eigenvalues and eigenvectors: $A (v_R \pm j v_I) = (\lambda_R \pm j\lambda_I)( v_R \pm jv_I)$, where $j=\sqrt{-1}$. We can rewrite it as
   
$$
A[v_R, v_I]= [v_R, v_I]  \begin{bmatrix}
\lambda_R & \lambda_I \\
-\lambda_I & \lambda_R 
\end{bmatrix}
$$

This sugguests that we always can block diagonalize $A$, and those $2\times 2$ blocks will have either of the following forms

$${\rm real \\, modes:}\\; \begin{bmatrix}
\lambda_1 & 0 \\
0 & \lambda_2 
\end{bmatrix}, \quad {\rm complex \\, modes:}\\; \begin{bmatrix}
\lambda_R & \lambda_I \\
-\lambda_I & \lambda_R 
\end{bmatrix}$$

There are some rough notes [here](https://www.overleaf.com/read/wwjbsyjsyfrm#f6fa3b).

### Some features

1) It supports sample rate conversion (see resample_up and resample_down settings).
2) It can enforce stability by pulling poles into the unit disc as 
   
   $$\lambda \rightarrow \lambda/\sqrt{|\lambda|^2 + 1}$$

    By default, we do not enforce stability (could be unstable for long sequences). 
  
3) May need to manually tune the scales of matrices $B$ and $C$ for optimal performance. Their default scales could be too large for long sequences. 

## Comparison against Mamba and Attention
On a simple language problem we compare the training loss perplexities of our Complex SSM vs Real vs Mamba vs Attention

![image](https://github.com/user-attachments/assets/ba60d266-59e8-43de-bee8-ac02b37571d6)
