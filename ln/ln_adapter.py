import torch
import torch.nn.functional as F
from torch import nn

class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # Group normalization
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6)
        # Query, key and value mappings
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        # Final $1 \times 1$ convolution layer
        self.proj_out = nn.Conv2d(channels, channels, 1)
        # Attention scaling factor
        self.scale = channels ** -0.5
        self._init_()

    def _init_(self):
        # initialize the weights
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum('bij,bcj->bci', attn, v)

        # Reshape back to `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w)
        # Final $1 \times 1$ convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out

class MyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # downsampling convolution
        # self.range_min = SmallRefinement()
        self.downconv = nn.Conv2d(4, 128, 3, stride=2, padding=1)
        self.attention = AttnBlock(128)
        self.upconv = nn.ConvTranspose2d(128, 4, 3, stride=2, padding=1, output_padding=1)
        self._init_()

    def _init_(self):
        # initialize the weights
        nn.init.xavier_uniform_(self.downconv.weight)
        nn.init.xavier_uniform_(self.upconv.weight)

    def forward(self, x):
        # Forward pass through each layer
        # x = self.range_min(x)
        x = x * 50 -30
        skip = x
        x = self.downconv(x)
        x = self.attention(x)
        x = self.upconv(x)
        x = x + skip
        return x

class SmallRefinement(nn.Module):
    def __init__(self):
        super(SmallRefinement, self).__init__()
        # add parameter mean and std to the model
        self.register_parameter('range', torch.nn.Parameter(torch.FloatTensor([50])))
        self.register_parameter('min', torch.nn.Parameter(torch.FloatTensor([-30])))
    def forward(self, x):
        # Forward pass through each layer
        x = x * self.range + self.min
        return x

# net = MyAdapter()
# print number of parameters
# print("Number of parameters:", sum(p.numel() for p in net.parameters()))
