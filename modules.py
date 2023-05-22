import torch
import torch.nn as nn
import torch.nn.functional as F




class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.stddev == 0.0: return din
        if self.training:
            return din + torch.randn(din.size(),device=din.device) * self.stddev
        return din


def hidden_block(dim_hidden,dim_hidden2,noise_sd,kernel_size=3,padding=1):
    return nn.Sequential(
        nn.GroupNorm(1,dim_hidden),
        GaussianNoise(stddev=noise_sd),
        nn.Conv2d(dim_hidden,dim_hidden2,kernel_size=(kernel_size,kernel_size),padding=padding,stride=1,bias=False),
    )
class ResidualBlock(nn.Module):
    def __init__(self,dim_hidden,dim_hidden2,noise_sd,dropout=0.0,kernel_size=3,padding=1):
        super().__init__()
        self.dropout=dropout
        self.block1=hidden_block(dim_hidden,dim_hidden2,noise_sd,kernel_size=kernel_size,padding=padding)
        self.block2=hidden_block(dim_hidden2,dim_hidden2,noise_sd,kernel_size=kernel_size,padding=padding)
        self.block3=hidden_block(dim_hidden2,dim_hidden2,noise_sd=0.0,kernel_size=kernel_size,padding=padding)
        self.bottleneck=nn.Conv2d(dim_hidden,dim_hidden2,kernel_size=(1,1),padding=0)

    def forward(self,x):
        identity=x
        out=F.leaky_relu(self.block1(x),0.01)
        out=F.dropout2d(out,self.dropout)
        out=F.leaky_relu(self.block2(out),0.01)
        out = F.dropout2d(out, self.dropout)
        out=self.block3(out)
        out = out + self.bottleneck(identity)
        out= F.leaky_relu(out,0.01)
        return out



class MLPModel(nn.Module):
    def __init__(self,dropout=0.01,hidden_dim=32,dim_out=1,normalization_groups=0,gaussian_sd=0,n_blocks=2,expansion=2):
        super().__init__()
        self.dropout=dropout
        if normalization_groups==0:
            self.input_norm=nn.Identity()
        else:
            self.input_norm=nn.GroupNorm(normalization_groups,12)
        self.conv1=nn.Conv2d(12,hidden_dim,kernel_size=(1,1),stride=1,padding=0)
        # self.bn1=nn.BatchNorm2d(hidden_dim)
        block_sizes=[hidden_dim]
        for i in range(n_blocks): block_sizes.append(block_sizes[i]*expansion)
        self.blocks=nn.Sequential(*[ResidualBlock(block_sizes[l-1],block_sizes[l],gaussian_sd) for l in range(1,len(block_sizes))])
        self.fc=nn.Linear(block_sizes[-1],dim_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,x):
        # x=x.sqrt()
        x=self.input_norm(x)
        x = F.dropout2d(x, self.dropout)
        x=F.leaky_relu(self.conv1(x),0.01)
        x=self.blocks(x)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x,1)
        x=F.dropout(x,p=self.dropout)
        x = self.fc(x)
        return x

class MLPModel2(nn.Module):
    def __init__(self,dropout=0.01,hidden_dim=32,dim_out=1,normalization_groups=0,gaussian_sd=0,n_blocks=2,expansion=2):
        super().__init__()
        self.dropout=dropout
        if normalization_groups==0:
            self.input_norm=nn.Identity()
        else:
            self.input_norm=nn.GroupNorm(normalization_groups,12)
        self.conv1=nn.Conv2d(12,hidden_dim,kernel_size=(1,1),stride=1,padding=0)
        # self.bn1=nn.BatchNorm2d(hidden_dim)
        block_sizes=[hidden_dim]
        for i in range(n_blocks): block_sizes.append(block_sizes[i]*expansion)
        self.blocks=nn.Sequential(*[ResidualBlock(block_sizes[l-1],block_sizes[l],gaussian_sd) for l in range(1,len(block_sizes))])
        # self.fc=nn.Linear(block_sizes[-1],dim_out)
        self.conv_final=nn.Conv2d(block_sizes[-1],dim_out,kernel_size=(1,1),stride=1,padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,x):
        # x=x.sqrt()
        x=self.input_norm(x)
        x = F.dropout2d(x, self.dropout)
        x=F.leaky_relu(self.conv1(x),0.01)
        x=self.blocks(x)
        x=self.conv_final(x)
        x=torch.mean(x,dim=(2,3))
        # x = F.adaptive_avg_pool2d(x,(1,1))
        # x = torch.flatten(x,1)
        # x=F.dropout(x,p=self.dropout)
        # x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self,dropout=0.01,hidden_dim=32,dim_input=12,dim_out=1,normalization_groups=0,gaussian_sd=0,n_blocks=1,expansion=2,kernel_size=3):
        super().__init__()
        if kernel_size ==1:
            padding=0
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size ==7:
            padding=3
        else:
            raise Exception(f"CNN not implemented for kernel_size{kernel_size}")
        self.dropout=dropout
        if normalization_groups==0:
            self.input_norm=nn.Identity()
        else:
            self.input_norm=nn.GroupNorm(normalization_groups,12)
        self.conv1=nn.Conv2d(dim_input,hidden_dim,kernel_size=(kernel_size,kernel_size),stride=1,padding=padding,bias=False)
        block_sizes=[hidden_dim]
        for i in range(n_blocks): block_sizes.append(block_sizes[i]*expansion)
        self.blocks=nn.Sequential(*[ResidualBlock(block_sizes[l-1],block_sizes[l],gaussian_sd,kernel_size=kernel_size,padding=padding)
                                    for l in range(1,len(block_sizes))])
        self.conv_final=nn.Conv2d(block_sizes[-1],dim_out,kernel_size=(kernel_size,kernel_size),stride=1,padding=padding)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,x):
        x=self.input_norm(x)
        x = F.dropout2d(x, self.dropout)
        x=F.leaky_relu(self.conv1(x),0.1)
        x=self.blocks(x)
        x=self.conv_final(x)
        x=torch.mean(x,dim=(2,3))
        return x