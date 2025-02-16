import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


"""
The Variational AutoEncoder (VAE) maps an input image to a 'small' latent space,
then reconstruct the image from the latent representation.
Original Image -> Encoder -> latent space representation -> Decoder -> reconstructed image 
"""

class VAE_Encoder(nn.Sequential): #inherits from nn.Sequential -> basically our class is a sequence of submodules

    def __init__(self):
        """
            keep diminuishing the dimension of the image but increase the channels,
            each pixel contains more info.
        """
        super().__init__(

            # (batch_size, channels, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            # stride=2 -> skip every two
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height, width) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            # (batch_size, 256, height, width) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512,512),

            nn.GroupNorm(32,512),

            nn.SiLU(),

            # here we decrease the number of features
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x:     (batch_size, channels, height, width)
        noise: (batch_size, output_channels, height/8, width/8)
        """
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # for asimmetrical padding
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (batch_size, 8, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        #                                    -> (batch_size, 4, height/8, width/8)
        # splits it into 2 tensors on the 1st dimension
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Z = N(0, 1) -> N(mean, variance) = X ?
        # X = mean + stdev * Z
        X = mean + stdev * noise

        # scale the output by a constant
        x *= 0.18125

        return x




class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)



class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # like layer norm but grouped together.
        # normalization -> We don't want them to oscillate too much
        # -> the loss will oscillate a lot
        # -> the training will be slower
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, features, height, width)
        
        residue = x 
        
        n, c, h, w = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height*width)
        x = x.view((n, c, h*w))

        # (batch_size, features, height*width) -> (batch_size, height*width, features)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2)

        # (batch_size, features, height*width) -> (batch_size, features, height, width)
        x = x.view((n, c, h, w))

        x += residue

        return x



# Here we have to return to the dimension of the original image
# With Encoder we went down, now we go back again
class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(

            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # height and width doubles...
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 512, height/2, width/2) -> (batch_size, 512, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, 512, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, 4, height/8, width/8)

        x /= 0.18125

        for module in self:
            x = module(x)

        # (batch_size, 3, height, width)
        return x
        

# It's basically our U-Net
class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        """
            latent -> (batch_size, 4, height/8, width/8)
            context -> (batch_size, seq_len, dim)
            time -> (1, 320)
        """
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        output = self.final(output)

        # (batch_size, 4, height/8, width/8)
        return output



class TimeEmbedding(nn.Module):

    def __init__(self, n_emb: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_emb, 4*n_emb)
        self.linear_2 = nn.Linear(4*n_emb, 4*n_emb)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1, 320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        return x



class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # keep decreasing the size of the image 

            # (batch_size, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (batch_size, 320, height/8, width/8) -> (batch_size, 320, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (batch_size, 640, height/16, width/16) -> (batch_size, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, height/32, width/32) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280))

        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            # here we do the opposite of the self.encoders
            # the input we expect is double the size because of the skip connection

            # (batch_size, 2560, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40))

        ])
    

    def forward(self, x, context, time):
        # x: (batch_size, 4, h / 8, w / 8)
        # context: (batch_size, seq_len, dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x




class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    

    def forward(self, feature, time): # feature is the latent
        """
            feature: (batch_size, in_channels, height, weight)
            time: (1, 1280)
        """

        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        # time doesn't have the batch_size and in_channels so we add them
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        merged += self.residual_layer(residue)

        return merged



class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4* channels *2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    

    def forward(self, x, context):
        """
            x is the latent (batch_size, features, height, width)
            context is the prompt (batch_size, seq_len, dim)
        """

        residue_long = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height*width)
        x = x.view(n, c, h*w)
        
        # (batch_size, features, height*width) -> (batch_size, height*width, features)
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection

        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x

        # Normalization + Cross Attention with skip connection

        x = self.layernorm_2(x)
        x = self.attention_2(x, context)

        x += residue_short

        residue_short = x

        # Normalization + FF with GeGLU and skip connection

        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

        x = self.conv_output(x) + residue_long

        return x



class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    

    def forward(self, x):
        # (batch_size, features, height, width) -> (batch_size, features, height*2, width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x



class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)
        
        # (batch_size, 4, height/8, width/8)
        return x



class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x



