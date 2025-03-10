import numpy as np
import torch

def fast_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


class StateSpaceModel(torch.nn.Module):
    """
    It returns a fully trainable state space model defined as:
        
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
                 has_matrixD: bool=False, has_bias: bool=False, decimation: int=1) -> None:
        """
        Inputs:
            input_size, state_size and output_size: sizes of u, x, and y, respectively.
            has_matrixD: matrix D is None if setting to False, otherwise not. 
            has_bias: bias b is None if setting to False, otherwise not. 
            decimation: decimate states and outputs if > 1 (same as the stride parameter in CNN),
                        interpolate states if < 1 (ToDo; similar to tranposed CNN). 
        """
        super(StateSpaceModel, self).__init__()
        A = 2*torch.pi*torch.rand(state_size)
        A = torch.complex(torch.cos(A), torch.sin(A))
        self.A = torch.nn.Parameter(0.9999*A) # slightly attentuate A to avoid numerical error accumulation in FFT 
        self.B = torch.nn.Parameter(
            torch.randn(input_size, state_size, dtype=torch.complex64)/(state_size + input_size)**0.5)
        self.C = torch.nn.Parameter(
            torch.randn(state_size, output_size, dtype=torch.complex64)/state_size**0.5)
        self.has_matrixD = has_matrixD
        if has_matrixD:
            self.D = torch.nn.Parameter(torch.zeros(input_size, output_size))
        self.has_bias = has_bias
        if has_bias:
            self.b = torch.nn.Parameter(torch.zeros(output_size))
        self.decimation = decimation


    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):
        """
        Inputs:
            u: the real input tensor with shape [batch, length, input_size].
            x0: the complex initial state with shape [batch, state_size].
            
        Outputs:
            y: the real output tensor with shape [batch, length, output_size].
            x0: the complex final state with shape [batch, state_size]. 
        """            
        _, length, _ = u.shape
        A_powers = torch.pow(self.A, torch.arange(length, device=self.A.device)[:, None])
        uB = u.to(torch.complex64) @ self.B # tranpose of math eq B*u
        x = fast_conv(A_powers, uB)[:, :length]
        if x0 is not None:
            x = x + (self.A * A_powers) * x0[:,None,:]
        if self.decimation > 1: # decimate to length length//decimation
            x = x[:, self.decimation-1::self.decimation]
            if self.has_matrixD:
                u = u[:, self.decimation-1::self.decimation]
        y = torch.real(x @ self.C) # tranpose of math eq C * x
        if self.has_matrixD:
            y = y + u @ self.D # transpose of math eq D * u 
        if self.has_bias:
            y = y + self.b
        return (y, x[:, -1])
        

class RealStateSpaceModel(torch.nn.Module):
    """
    The same as the above StateSpaceModel except that everything is real.
    Limited representation capacity, not very useful. Use it as a counterexample. 
    """
    def __init__(self, input_size: int, state_size: int, output_size: int) ->None:
        super(RealStateSpaceModel, self).__init__()
        # training is unstable if initializing A to diag(random +1 or -1)
        self.A = torch.nn.Parameter(0.9999*(2*torch.rand(state_size) - 1)) 
        self.B = torch.nn.Parameter(torch.randn(input_size, state_size)/(state_size + input_size)**0.5)
        self.C = torch.nn.Parameter(torch.randn(state_size, output_size)/state_size**0.5)

    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):         
        _, length, _ = u.shape
        A_powers = torch.pow(self.A, torch.arange(length, device=self.A.device)[:, None])
        uB = u @ self.B
        x = torch.real(fast_conv(A_powers, uB)[:, :length])
        if x0 is not None:
            x = x + (self.A * A_powers) * x0[:,None,:]
        y = x @ self.C
        return (y, x[:, -1])


if __name__ == "__main__":    
    # test fast_conv
    print("Let's test function fast_conv")
    len1, len2 = 11, 17
    batch, i = 13, 5
    a = torch.randn(len1, i)
    b = torch.randn(batch, len2, i)
    c = fast_conv(a, b)
    c_forloop = torch.zeros(batch, len1 + len2 - 1, i)
    for m in range(len1):
        for n in range(len2):
            c_forloop[:, m + n] += a[m] * b[:, n] 
    print(f"Max errors between fast_conv and for_loop conv is {torch.max(torch.abs(c - c_forloop))}\n")
    
    # test the state space model class.
    print("Let's test class StateSpaceModel")
    input_size, state_size, output_size = 3, 7, 5
    ssm = StateSpaceModel(input_size, state_size, output_size, 
                          has_matrixD=True, has_bias=True, decimation=1)
    length = 10 # simply set batch size to 1 
                # if you set to a huge sequence length, numerical error eventually accumulates if max(abs(A)) is too close to 1
    u = torch.randn(length, input_size)
    x0 = torch.randn(state_size, dtype=torch.complex64)
    y, new_state = ssm(u[None,:,:], x0[None,:])
    
    # recursive implementation with for loop  
    state = x0
    for t in range(length):
        state = state * ssm.A + (u[t].to(torch.complex64)) @ ssm.B
        out = torch.real(state @ ssm.C) + u[t] @ ssm.D + ssm.b
        print(f"Max output err between conv and recursive views at step {t+1}: {torch.max(torch.abs(y[:,t] - out))}")
    print(f"Max final state err between conv and recursive views: {torch.max(torch.abs(new_state[0] - state))}")
    