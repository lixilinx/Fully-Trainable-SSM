# Complex-state-SSM
This is my simple fully trainable state space model (SSM) implementation. In most works, the state matrix ($A$) is fixed to certain ad hoc ones. Actually, it is the most important matrix in an SSM accounting for the long term memories. We should learn it from data! To achieve that, we need to turn to the eigenvalue decomposition (EVD) of $A$, and generally the state vectors can only be complex in an EVD form of SSM. Not a surprise as the EVD only always exists in the domain of complex numbers, even for real matrices. 

I also put a simple demo to show that SSMs with complex states do learn well.  



## Comparison against Mamba and Attention
On a simple language problem we compare the training loss perplexities of our Complex SSM vs Mamba and Attention

![image](https://github.com/user-attachments/assets/a011bef1-3684-4420-849b-8cddd7bfed3d)
