import torch
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor
from entmax import sparsemax
from entmax import normmax_bisect

########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks with learnable connections.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            gumbel_tau: float = 1,
            gate_function: str = 'softmax',
            connection_tau: float = 0.5,
            hard_connections: bool = False,
            num_connections: int = 16,
            noise_temp: float = 0.1,
            entmax_alpha: float = 1.5,
            residual: bool = False
    ):
        """
        :param in_dim: input dimensionality of the layer
        :param out_dim: output dimensionality of the layer
        :param device: device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), increase to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python')
        :param connections: method for initializing connectivity ('random', 'unique', 'learned')
        :param gumbel_tau: temperature for Gumbel-Softmax
        :param gate_function: gate selection method ('softmax', 'gumbel_softmax', 'sparsemax', 'gumbel_sparsemax', 'sparsemax_noise')
        :param connection_tau: temperature for connection selection
        :param hard_connections: use straight-through estimator for connections
        :param num_connections: number of candidate connections per neuron (for learned connections)
        :param noise_temp: temperature for Gaussian noise (for sparsemax_noise)
        """
        super().__init__()
        if not residual:
            self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        else:
            tens = torch.full((out_dim,16),0.1/15,device=device)
            tens[:,3] = 0.9
            self.weights = torch.nn.Parameter(tens)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.gumbel_tau = gumbel_tau
        self.gate_function = gate_function
        self.connection_tau = connection_tau
        self.hard_connections = hard_connections
        self.num_connections = min(num_connections, in_dim)
        self.noise_temp = noise_temp
        self.entmax_alpha = entmax_alpha
        self.implementation = implementation
        self.residual = residual
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.weights_logvar = torch.nn.parameter.Parameter(torch.zeros(out_dim, 16, device=device))
        torch.nn.init.constant_(self.weights_logvar, -10)

        self.connections = connections
        assert self.connections in ['random', 'unique', 'learned'], self.connections
        
        if self.connections == 'learned':
            # Initialize candidate input indices for each output neuron
            candidate_indices = torch.zeros(self.out_dim, self.num_connections, dtype=torch.int64, device=device)
            for i in range(self.out_dim):
                candidates = torch.randperm(self.in_dim, device=device)[:self.num_connections]
                candidate_indices[i] = candidates
            self.register_buffer("candidate_indices", candidate_indices)
            
            # Learnable weights to select among candidates
            self.connection_logits_a = torch.nn.parameter.Parameter(
                torch.randn(out_dim, self.num_connections, device=device) * 0.1
            )
            self.connection_logits_b = torch.nn.parameter.Parameter(
                torch.randn(out_dim, self.num_connections, device=device) * 0.1
            )
            
            self.in_conn = None
            self.out_conn = None
            self.indices = None
        else:
            # Fixed connections
            a, b = self.get_connections(self.connections, device)
            self.register_buffer("in_conn", a)
            self.register_buffer("out_conn", b)
            self.indices = (a, b)
            self.connection_logits_a = None
            self.connection_logits_b = None

        # Setup CUDA backward indices for fixed connections only
        if self.implementation == 'cuda' and self.connections != 'learned':
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), 
                device=device, dtype=torch.int64
            )
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], 
                dtype=torch.int64, device=device
            )

        self.num_neurons = out_dim
        self.num_weights = out_dim


    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA.'

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)


    def get_soft_connections(self):
        """Compute soft connection weights using Gumbel-Softmax or Sparsemax."""
        if self.gate_function == 'gumbel_softmax':
            weights_a = torch.nn.functional.gumbel_softmax(
                self.connection_logits_a, 
                tau=self.connection_tau, 
                hard=self.hard_connections,
                dim=-1
            )
            weights_b = torch.nn.functional.gumbel_softmax(
                self.connection_logits_b, 
                tau=self.connection_tau, 
                hard=self.hard_connections,
                dim=-1
            )
        elif self.gate_function == 'sparsemax':
            weights_a = sparsemax(self.connection_logits_a / self.connection_tau, dim=-1)
            weights_b = sparsemax(self.connection_logits_b / self.connection_tau, dim=-1)
        else:
            weights_a = torch.nn.functional.softmax(self.connection_logits_a / self.connection_tau, dim=-1)
            weights_b = torch.nn.functional.softmax(self.connection_logits_b / self.connection_tau, dim=-1)
        
        return weights_a, weights_b


    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)
        
        if self.connections == 'learned':
            weights_a, weights_b = self.get_soft_connections()
            # Gather candidate inputs: [batch, out_dim, num_connections]
            x_candidates = x[:, self.candidate_indices]
            # Weighted sum over candidates: [batch, out_dim]
            a = (x_candidates * weights_a.unsqueeze(0)).sum(dim=-1)
            b = (x_candidates * weights_b.unsqueeze(0)).sum(dim=-1)
        else:
            if self.indices[0].dtype != torch.int64:
                self.indices = self.indices[0].long(), self.indices[1].long()
            a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        
        # Gate logic with all optimizer options
        if self.training:
            if self.gate_function == 'gumbel_softmax':
                w = torch.nn.functional.gumbel_softmax(
                    self.weights, dim=-1, tau=self.gumbel_tau, hard=False
                )
            elif self.gate_function == 'softmax':
                w = torch.nn.functional.softmax(self.weights, dim=-1)
            elif self.gate_function == 'sparsemax':
                w = sparsemax(self.weights, dim=-1)
            elif self.gate_function == 'sparsemax_noise':
                noise = self._sample_gaussian_noise(self.weights, self.noise_temp)
                noisy_logits = self.weights + noise
                w = sparsemax(noisy_logits, dim=-1)
            elif self.gate_function == 'gumbel_sparsemax':
                tau = getattr(self, 'gumbel_tau', 0.25)
                g = self._sample_gumbel_like(self.weights, eps=1e-10)
                logits = (self.weights + g) / tau
                w = sparsemax(logits, dim=-1)
                if getattr(self, 'hard', False):
                    w_hard = torch.nn.functional.one_hot(
                        w.argmax(-1), w.size(-1)
                    ).to(w.dtype)
                    w = (w_hard - w).detach() + w
            else:
                raise ValueError(f"Invalid gate function: {self.gate_function}")
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
        
        x = bin_op_s(a, b, w)
        return x


    @staticmethod
    def _sample_gumbel_like(t, eps=1e-10):
        """Sample Gumbel noise for Gumbel-Softmax/Sparsemax."""
        U = torch.rand_like(t).clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(U))


    @staticmethod
    def _sample_gaussian_noise(t, temp=1.0):
        """Sample Gaussian noise scaled by temperature."""
        return torch.randn_like(t) * temp


    def forward_cuda(self, x):
        if self.connections == 'learned':
            return self.forward_python(x)
        
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, f"{x.shape}"
        x = x.transpose(0, 1).contiguous()
        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            if self.gate_function == 'gumbel_softmax':
                w = torch.nn.functional.gumbel_softmax(
                    self.weights, dim=-1, tau=self.gumbel_tau, hard=False
                ).to(x.dtype)

            elif self.gate_function == 'softmax':
                w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)

            elif self.gate_function == 'sparsemax':
                w = sparsemax(self.weights, dim=-1).to(x.dtype)

            elif self.gate_function == 'sparsemax_noise':
                noise = self._sample_gaussian_noise(self.weights, self.noise_temp)
                noisy_logits = self.weights + noise
                w = sparsemax(noisy_logits, dim=-1).to(x.dtype)
            
            elif self.gate_function == 'log_softmax':
                log_w = torch.nn.functional.log_softmax(self.weights, dim=-1)
                w = torch.exp(log_w).to(x.dtype)

            elif self.gate_function == 'gumbel_sparsemax':
                tau = getattr(self, 'gumbel_tau', 0.25)
                g = self._sample_gumbel_like(self.weights, eps=1e-10)
                logits = (self.weights + g) / tau
                w = sparsemax(logits, dim=-1).to(x.dtype)
                if getattr(self, 'hard', False):
                    w_hard = torch.nn.functional.one_hot(
                        w.argmax(-1), w.size(-1)
                    ).to(w.dtype)
                    w = (w_hard - w).detach() + w

            elif self.gate_function == 'entmax':
                w = normmax_bisect(self.weights, alpha=self.entmax_alpha,dim=-1)

            elif self.gate_function == 'relu_normalized':
                temp = self.weights ** 2
                relu_out = torch.nn.functional.relu(temp)
                w = (relu_out / (relu_out.abs().sum(dim=-1, keepdim=True) + 1e-10)).to(x.dtype)

            elif self.gate_function == 'balenas':
                eps = torch.randn_like(self.weights)
                std = torch.exp(0.5*self.weights_logvar)
                alpha_sample = self.weights + eps * std
                w = torch.nn.functional.softmax(alpha_sample,dim=-1).to(x.dtype)

            else:
                raise ValueError("Invalid gate function option")
                
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)


    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        if self.connections == 'learned':
            # Discretize: pick highest weight among candidates
            selected_a = self.connection_logits_a.argmax(dim=-1)
            selected_b = self.connection_logits_b.argmax(dim=-1)
            # Map to actual input indices
            a = self.candidate_indices.gather(1, selected_a.unsqueeze(-1)).squeeze(-1)
            b = self.candidate_indices.gather(1, selected_b.unsqueeze(-1)).squeeze(-1)
        else:
            a, b = self.indices
        
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)
        return x


    def extra_repr(self):
        conn_type = 'learned' if self.connections == 'learned' else 'fixed'
        return f'{self.in_dim}, {self.out_dim}, {conn_type}, {"train" if self.training else "eval"}'


    def get_connections(self, connections, device='cuda'):
    # Modified to handle cases where out_dim < in_dim/2 (compression layers)
        if connections == 'random':
            # Each output neuron needs 2 inputs
            # When compressing (out_dim < in_dim/2), we can still randomly sample pairs
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = c.reshape(2, self.out_dim)
            a, b = c[0].to(torch.int64).to(device), c[1].to(torch.int64).to(device)
            return a, b
        elif connections == 'unique':
            # For unique connections, we need out_dim * 2 <= in_dim
            # If compression is too aggressive, fall back to random with reuse
            if self.out_dim * 2 > self.in_dim:
                print(f"Warning: Cannot generate unique connections for out_dim={self.out_dim}, in_dim={self.in_dim}. Using random connections.")
                return self.get_connections('random', device)
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

    

    def get_connection_entropy(self):
        """Compute entropy of connection distributions for regularization."""
        if self.connections != 'learned':
            return 0.0
        
        weights_a = torch.nn.functional.softmax(self.connection_logits_a / self.connection_tau, dim=-1)
        weights_b = torch.nn.functional.softmax(self.connection_logits_b / self.connection_tau, dim=-1)
        
        entropy_a = -(weights_a * torch.log(weights_a + 1e-10)).sum(dim=-1).mean()
        entropy_b = -(weights_b * torch.log(weights_b + 1e-10)).sum(dim=-1).mean()
        
        return (entropy_a + entropy_b) / 2
    
    def get_kl_loss(self):
        if self.gate_function != 'balenas':
            return 0.0
        kl = -0.5 * torch.sum(
        1 + self.weights_logvar - self.weights.pow(2) - self.weights_logvar.exp()
        )
        return kl


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        """
        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################
