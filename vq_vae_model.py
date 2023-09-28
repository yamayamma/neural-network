import torch
import torch.nn as nn
import torch.nn.functional as F

# Decoder out_channels= 3 or 1 , color:mono

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost): # K, D, beta
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings) # 一様分布
        self._commitment_cost = commitment_cost # 0.25 ?

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous() # 軸の入れ替え　参照ではない
        input_shape = inputs.shape
        
        # Flatten input (16384, 64)
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances 2乗距離 (16384, 512)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self._embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding　One-hot表現
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # 最短距離のindex
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # Tensor.scatter_(dim, index, src, reduce=None)
        
        # Quantize and unflatten 埋め込みベクトルに置き換え
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized2 = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # 第2項
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # 第3項
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach() # 勾配の切り離し
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, distances ,encoding_indices, encodings

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens): # 128, 128, 32
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False), # チャンネル数32に減らす
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False) # チャンネル数128に戻す
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):# 128, 128, 2, 32
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)]) # residual 2回

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens): # 3, 128, 2, 32
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1) # チャンネル数64に増やす サイズ1/2
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1) # チャンネル数128に増やす サイズ1/2
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1) # 維持
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens): # D, 128, 2, 32
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1) # D, 128
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens) # 128, 128, 2, 32
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1) # 128, 64 サイズ2倍
        # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=3, kernel_size=4, stride=2, padding=1) # 64, channel数 サイズ2倍(元のサイズ)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)

class Model(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens) # 3, 128, 2, 32
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1) # 128, 64
        # if decay > 0.0:
        #     self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
        #                                       commitment_cost, decay)
        # else:
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost) # K, D, beta
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens) # D, 128, 2, 32

    def forward(self, x):
        z = self._encoder(x) # channel=128, size=1/4
        z = self._pre_vq_conv(z) # channel=64 埋め込みベクトルの次元にする
        loss, quantized, perplexity, distances, encoding_indices, encodings = self._vq_vae(z) # 埋め込みベクトルに置き換え
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity, distances, encoding_indices, encodings
