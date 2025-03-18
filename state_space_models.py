import numpy as np
import torch

########################### Complex State 1D SSM #################################

def fast_conv_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
        a: tensor with shape [len1, n] representing len1 n x n diagonal matrices.
        b: another torch tensor with shape [batch, len2, n].
        
    Outputs:
        c = conv(a, b) with shape [batch, len1 + len2 - 1, n]. 
    """
    len1, _ = a.shape
    _, len2, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = torch.fft.fft(a, n=T, dim=0)
    B = torch.fft.fft(b, n=T, dim=1)
    C = A * B
    c = torch.fft.ifft(C, n=T, dim=1)
    return c[:, :(len1 + len2 - 1)]


class ComplexStateSpaceModel(torch.nn.Module):
    """
    It returns a fully trainable complex state SSM defined as:
        
        x_t = A @ x_{t-1} + B @ u_t,
        y_t = C @ x_t + D @ u_t + b,
        
    where:
        
        u_t, 1 <= t <= T, are the sequences of (real) inputs,
        x_t, 1 <= t <= T, are the sequence of (complex) states,
        y_t, 1 <= t <= T, are the sequence of (real) outputs,
        matrices A (diagonal complex), B (complex) and C (complex) are mandatory, 
        matrix D (real) and bias b (real) are optional,
        and x_0 is the initial (complex) state. 
        
    Note that we use the tranposed version of these equations in the Python code 
    by following the 'row major' convention.  
    """ 
    def __init__(self, input_size: int, state_size: int, output_size: int, 
                 has_matrixD: bool=False, has_bias: bool=False, 
                 resample_up: int=1, resample_down: int=1, 
                 enforce_stability: bool=False) -> None:
        """
        Inputs:
            input_size, state_size and output_size: sizes of u, x, and y, respectively.
            has_matrixD: matrix D is None if setting to False, otherwise not. 
            has_bias: bias b is None if setting to False, otherwise not. 
            resample_up, resample_down: resample the sequence with ratio resample_up/resample_down. 
            enforce_stability: set to True to enforce the poles to stay inside unit disc, 
                        otherwise poles can be outside of unit disc (unstable for long sequences).  
        """
        super(ComplexStateSpaceModel, self).__init__()
        A = 2*torch.pi*torch.rand(state_size)
        A = torch.complex(torch.cos(A), torch.sin(A))
        self.enforce_stability = enforce_stability
        if enforce_stability:
            self.A = torch.nn.Parameter(10 * A)
        else:
            self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(
            torch.randn(input_size, state_size, dtype=torch.complex64)/(state_size + input_size)**0.5)
        self.C = torch.nn.Parameter(
            torch.randn(state_size, output_size, dtype=torch.complex64)/(state_size + output_size)**0.5)
        self.has_matrixD = has_matrixD
        if has_matrixD:
            self.D = torch.nn.Parameter(torch.zeros(input_size, output_size))
        self.has_bias = has_bias
        if has_bias:
            self.b = torch.nn.Parameter(torch.zeros(output_size))
        self.resample_up = resample_up
        self.resample_down = resample_down


    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):
        """
        Inputs:
            u: the real input tensor with shape [batch, length, input_size].
            x0: the complex initial state with shape [batch, state_size].
            
        Outputs:
            y: the real output tensor with shape [batch, resample_up*length//resample_down, output_size].
            x0: the complex final state with shape [batch, state_size]. 
        """    
        if self.resample_up > 1:
            u = u.repeat_interleave(self.resample_up, 1)
        _, length, _ = u.shape
        if self.enforce_stability:
            A = self.A * torch.rsqrt(self.A * self.A.conj() + 1) # pull inside the unit disc  
        else:
            A = self.A 
        Aps = torch.pow(A, torch.arange(length + 1, device=self.A.device)[:, None])
        uB = u.to(torch.complex64) @ self.B # tranpose of math eq B*u
        x = fast_conv_complex(Aps[:-1], uB)[:, :length]
        if x0 is not None:
            x = x + Aps[1:] * x0[:,None,:]
        if self.resample_down > 1: 
            x = x[:, self.resample_down-1::self.resample_down]
            if self.has_matrixD:
                u = u[:, self.resample_down-1::self.resample_down]
        y = torch.real(x @ self.C) # tranpose of math eq C * x
        if self.has_matrixD:
            y = y + u @ self.D # transpose of math eq D * u 
        if self.has_bias:
            y = y + self.b
        return (y, x[:, -1])
    
        
############################## Real State 1D SSM #################################


def fast_conv_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
        a: tensor with shape [len1, K, n, n] representing len1 x K diagonal matrices, 
           and each block has size n x n. 
        b: another torch tensor with shape [batch, len2, K, n] representing batch x len2 x K vectors, 
           and each has length n.
        
    Outputs:
        c = conv(b, a) with shape [batch, len1 + len2 - 1, K, n]. 
        Following the 'row major' convention, here we take the vector-matrix product (b x a). 
    """
    len1, _, _, _ = a.shape
    _, len2, _, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = torch.fft.rfft(a, n=T, dim=0)
    B = torch.fft.rfft(b, n=T, dim=1)
    C = torch.einsum("blki, lkij->blkj", B, A)
    c = torch.fft.irfft(C, n=T, dim=1)
    return c[:, :(len1 + len2 - 1)]


def A_powers(A: torch.Tensor, t: int) -> torch.Tensor:
    """
    Inputs:
        A: tensor with shape [K, n, n] representing a block diagonal matrice, 
           each block has size n x n, and a total of K such blocks.
        t: a positive integer.
        
    Outputs:
        torch.stack([I, A, A^2, ..., A^(t - 1)], dim=0), 
        a tensor with shape [t, K, n, n]. 
    """
    K, n, _ = A.shape 
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    result = eye.repeat(1, K, 1, 1)
    lift = A
    while len(result) < t:
        result = torch.cat([result, result @ lift])
        lift = lift @ lift
    return result[:t]


def shrink_eigs(A):
    """
    For a 2 x 2 matrix, we pull its largest eigenvalue into the unit disc as 
    
        lambda / sqrt(|lambda|^2 + 1)
        
    This function works only for 2 x 2 matrices. 
    """
    tr, det = A[...,0,0] + A[...,1,1], A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
    discriminant = tr*tr - 4*det
    max_rr = torch.where(discriminant<=0, 
                         det,   # a pair of complex eigenvalues; so |r|^2 = det 
                         (torch.abs(tr) + torch.sqrt(torch.abs(discriminant) + 1e-30))**2/4,
                         # two real eigenvalues; this gives max(r1^2, r2^2)
                         )
    sA = A * torch.rsqrt(max_rr[...,None,None] + 1)
    return sA


class RealStateSpaceModel(torch.nn.Module):
    """
    It returns a fully trainable real state SSM defined as:
        
        x_t = A @ x_{t-1} + B @ u_t,
        y_t = C @ x_t + D @ u_t + b,
        
    where:
        
        u_t, 1 <= t <= T, are the sequences of (real) inputs,
        x_t, 1 <= t <= T, are the sequence of (real) states,
        y_t, 1 <= t <= T, are the sequence of (real) outputs,
        matrices A (block diagonal real), B (real) and C (real) are mandatory, 
        matrix D (real) and bias b (real) are optional,
        and x_0 is the initial (real) state. 
        
    Note that we use the tranposed version of these equations in the Python code 
    by following the 'row major' convention.  
    """ 
    def __init__(self, input_size: int, state_size: int, output_size: int,  
                 has_matrixD: bool=False, has_bias: bool=False, 
                 resample_up: int=1, resample_down: int=1, 
                 enforce_stability: bool=False, state_blk_size: int=2) -> None:
        """
        Inputs:
            input_size, state_size and output_size: sizes of u, x, and y, respectively, 
                    and the state_size must be an even number.
            has_matrixD: matrix D is None if setting to False, otherwise not. 
            has_bias: bias b is None if setting to False, otherwise not. 
            resample_up, resample_down: resample the sequence with ratio resample_up/resample_down. 
            enforce_stability: set to True to enforce the poles to stay inside unit disc 
                            (currently only works for state block size 2), 
                            otherwise poles can be outside of unit disc (unstable for long sequences).  
            state_blk_size: an even integer for the block size of block diagonal matrix A.  
        """
        super(RealStateSpaceModel, self).__init__()
        assert(state_blk_size % 2 == 0)
        assert(state_size % state_blk_size == 0)
        self.state_blk_size = state_blk_size
        self.state_num_blks = state_size // state_blk_size
        theta = 2*torch.pi*torch.rand(state_size//2)
        A2 = torch.zeros(state_size//2, 2, 2) # 2 x 2 blocks by default 
        A2[:,0,0] = torch.cos(theta)
        A2[:,0,1] = torch.sin(theta)
        A2[:,1,0] = -torch.sin(theta)
        A2[:,1,1] = torch.cos(theta)
        if state_blk_size > 2:
            A = torch.zeros(self.state_num_blks, state_blk_size, state_blk_size)
            for i in range(self.state_num_blks):
                A[i] = torch.block_diag(*A2[i * state_blk_size//2 : (i + 1) * state_blk_size//2])
        else:
            A = A2
        self.enforce_stability = enforce_stability
        if enforce_stability:
            assert(state_blk_size == 2)
            self.A = torch.nn.Parameter(10 * A)
        else:
            self.A = torch.nn.Parameter(A) 
        self.B = torch.nn.Parameter(
            torch.randn(input_size, state_size)/(state_size + input_size)**0.5)
        self.C = torch.nn.Parameter(
            torch.randn(state_size, output_size)/(state_size + output_size)**0.5)
        self.has_matrixD = has_matrixD
        if has_matrixD:
            self.D = torch.nn.Parameter(torch.zeros(input_size, output_size))
        self.has_bias = has_bias
        if has_bias:
            self.b = torch.nn.Parameter(torch.zeros(output_size))
        self.resample_up = resample_up
        self.resample_down = resample_down


    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):
        """
        Inputs:
            u: the real input tensor with shape [batch, length, input_size].
            x0: the real initial state with shape [batch, state_num_blk, state_blk_size]. 
            
        Outputs:
            y: the real output tensor with shape [batch, resample_up*length//resample_down, output_size].
            x0: the real final state with shape [batch, state_num_blk, state_blk_size]. 
        """      
        if self.resample_up > 1:
            u = u.repeat_interleave(self.resample_up, 1)
        _, length, _ = u.shape
        if self.enforce_stability:
            A = shrink_eigs(self.A)
        else:
            A = self.A
        Aps = A_powers(A, length + 1)
        uB = u @ self.B # tranpose of math eq B*u
        uB = torch.reshape(uB, (-1, length, self.state_num_blks, self.state_blk_size))
        x = fast_conv_real(Aps[:-1], uB)[:, :length]
        if x0 is not None:
            x = x + torch.einsum("bki,lkij->blkj", x0, Aps[1:])
        new_state = x[:, -1]
        x = torch.reshape(x, (-1, length, self.state_num_blks * self.state_blk_size))
        if self.resample_down > 1: 
            x = x[:, self.resample_down-1::self.resample_down]
            if self.has_matrixD:
                u = u[:, self.resample_down-1::self.resample_down]
        y = x @ self.C # tranpose of math eq C * x
        if self.has_matrixD:
            y = y + u @ self.D # transpose of math eq D * u 
        if self.has_bias:
            y = y + self.b
        return (y, new_state)


if __name__ == "__main__":    
    # test fast_conv_complex
    print("Let's test function fast_conv_complex")
    len1, len2 = 11, 17
    batch, i = 13, 5
    a = torch.randn(len1, i, dtype=torch.complex64)
    b = torch.randn(batch, len2, i, dtype=torch.complex64)
    c = fast_conv_complex(a, b)
    c_forloop = torch.zeros(batch, len1 + len2 - 1, i, dtype=torch.complex64)
    for m in range(len1):
        for n in range(len2):
            c_forloop[:, m + n] += a[m] * b[:, n] 
    print(f"Max errors between fast_conv_complex and for_loop conv is {torch.max(torch.abs(c - c_forloop))}\n")
    
    # test the complex state SMM class.
    print("Let's test class ComplexStateSpaceModel")
    input_size, state_size, output_size = 3, 7, 5
    ssm = ComplexStateSpaceModel(input_size, state_size, output_size, 
                                 has_matrixD=True, has_bias=True)
    length = 10 # simply set batch size to 1 
                # if you set to a huge sequence length, numerical error eventually accumulates when max(abs(A)) is too close to 1
    u = torch.randn(length, input_size)
    x0 = torch.randn(state_size, dtype=torch.complex64)
    y, new_state = ssm(u[None,:,:], x0[None,:])
    
    # recursive implementation with for loop  
    state = x0
    for t in range(length):
        state = state * ssm.A + (u[t].to(torch.complex64)) @ ssm.B
        out = torch.real(state @ ssm.C) + u[t] @ ssm.D + ssm.b
        print(f"Max output err between conv and recursive views at step {t+1}: {torch.max(torch.abs(y[:,t] - out))}")
    print(f"Max final state err between conv and recursive views: {torch.max(torch.abs(new_state[0] - state))}\n")
    
    # test resampling 
    ssm = ComplexStateSpaceModel(input_size, state_size, output_size, 
                                 has_matrixD=True, has_bias=True, resample_up=3, resample_down=7)
    u = torch.randn(14, input_size)
    y, _ = ssm(u[None,:,:])
    print(f"Sequence lengths before and after resampling complex state SSM: {len(u)}, {y.shape[1]}\n")
    
    # test fast_conv_real
    print("Let's test function fast_conv_real")
    len1, len2 = 11, 17
    batch, K, n = 13, 5, 3
    a = torch.randn(len1, K, n, n)
    b = torch.randn(batch, len2, K, n)
    c = fast_conv_real(a, b)
    c_forloop = torch.zeros(batch, len1 + len2 - 1, K, n)
    for i in range(len2):
        for j in range(len1):
            for k in range(K):
                c_forloop[:, i + j, k] += b[:, i, k] @ a[j, k] 
    print(f"Max errors between fast_conv_real and for_loop conv is {torch.max(torch.abs(c - c_forloop))}\n")
    
    # test function A_powers
    print("Test function A_powers")
    K, n, t = 3, 2, 10
    A = torch.randn(K, n, n)
    powers = A_powers(A, t)
    powers_forloop = torch.eye(n).repeat(K, 1, 1)
    for i in range(t):
        print(f"Max err between fast and forloop powers of A at step {i+1}: {torch.max(torch.abs(powers[i] - powers_forloop))}")
        powers_forloop = powers_forloop @ A
    print("\n")
        
    # test the real state SMM class.
    print("Let's test class RealStateSpaceModel")
    input_size, state_size, output_size, state_blk_size = 3, 8, 5, 4
    ssm = RealStateSpaceModel(input_size, state_size, output_size, 
                              has_matrixD=True, has_bias=True, state_blk_size=state_blk_size)
    length = 10 # simply set batch size to 1 
                # if you set to a huge sequence length, numerical error eventually accumulates when max(abs(A)) is too close to 1
    u = torch.randn(length, input_size)
    x0 = torch.randn(state_size//state_blk_size, state_blk_size)
    y, new_state = ssm(u[None,:,:], x0[None,:])
    
    # recursive implementation with for loop  
    state = x0.view(-1)
    A = torch.block_diag(*ssm.A)
    for t in range(length):
        state = state @ A + u[t] @ ssm.B
        out = state @ ssm.C + u[t] @ ssm.D + ssm.b
        print(f"Max output err between conv and recursive views at step {t+1}: {torch.max(torch.abs(y[:,t] - out))}")
    print(f"Max final state err between conv and recursive views: {torch.max(torch.abs(new_state[0].view(-1) - state))}\n")
    
    # test resampling 
    ssm = RealStateSpaceModel(input_size, state_size, output_size, 
                              has_matrixD=True, has_bias=True, resample_up=3, resample_down=7)
    u = torch.randn(14, input_size)
    y, _ = ssm(u[None,:,:])
    print(f"Sequence length before and after resampling real state SSM: {len(u)}, {y.shape[1]}\n")
    
    # test function shrink_eigs
    for _ in range(10):
        A = torch.empty([]).exponential_() * torch.randn(2, 2)
        sA = shrink_eigs(A)
        print(f"Max |eigs| before and after shrinking: {torch.max(torch.abs(torch.linalg.eig(A)[0]))} --> {torch.max(torch.abs(torch.linalg.eig(sA)[0]))} ")