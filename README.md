# Fully Trainable SSM

Some experimental work on the state space model (SSM). It's in progress, and mainly for fun.  

In most works, the state matrix ($A$) is fixed, say the HiPPO matrix. Actually, it is the most important matrix in an SSM accounting for the long term memories. We should learn it from the data. There are two possible modal decomposition forms suitable for highly efficent fully trainable SSM learnings:

1) The eigenvalue decomposition (EVD) form. This is not difficult to implement as $A$ is diagonal. But the state vectors can be complex since the EVD only always exists in the domain of complex numbers, even for real matrices.

2) Another real number only modal decomposition forms is to make $A$ a block diagonal matrix with block size $2\times 2$. It is more difficult to train than the EVD form. But, everything stays in the domain of real numbers nicely.

 I have some very rough notes [here](https://www.overleaf.com/read/wwjbsyjsyfrm#f6fa3b).



## Comparison against Mamba and Attention
On a simple language problem we compare the training loss perplexities of our Complex SSM vs Mamba and Attention

![image](https://github.com/user-attachments/assets/a011bef1-3684-4420-849b-8cddd7bfed3d)
