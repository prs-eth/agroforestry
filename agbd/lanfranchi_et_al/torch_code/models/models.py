import torch
import torch.nn as nn
from torchsummary import summary
from models.unet_layers import *

class SimpleFCN(nn.Module):
    def __init__(self,
                 in_features=4,
                 out_features = (16, 32, 64, 128),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1,
                 max_pool=False):
        """
        A simple fully convolutional neural network.
        Args:
            in_features: input channel dimension (we give images with 4 channels).
            out_features: list of channel feature dimensions.
            num_outputs: number of the output dimension
        """
        super(SimpleFCN, self).__init__()
        self.relu = nn.ReLU(inplace=True) # inplace=True can sometimes slightly decrease the memory usage
        layers = list()
        for i in range(len(out_features)):
            in_channels = in_features if i == 0 else out_features[i-1]
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_features[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(num_features=out_features[i]))
            layers.append(self.relu)
            
            # c.f. https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
            # if max_pool = False, stride of Conv2d should be set to 2
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            
        self.conv_layers = nn.Sequential(*layers)
        
        self.conv_output = nn.Conv2d(in_channels=out_features[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        predictions = self.conv_output(x)
        return predictions

class SimpleFCN_Gaussian(nn.Module):
    def __init__(self,
                 in_features=4,
                 out_features = (16, 32, 64, 128),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1,
                 max_pool=False,
                 downsample=None):
        """
        A simple fully convolutional neural network.
        Args:
            in_features: input channel dimension (we give images with 4 channels).
            out_features: list of channel feature dimensions.
            num_outputs: number of the output dimension
        """
        super(SimpleFCN_Gaussian, self).__init__()
        self.relu = nn.ReLU(inplace=True) # inplace=True can sometimes slightly decrease the memory usage
        layers_mean = list()
        for i in range(len(out_features)):
            in_channels = in_features if i == 0 else out_features[i-1]
            layers_mean.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_features[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1))
            layers_mean.append(nn.BatchNorm2d(num_features=out_features[i]))
            layers_mean.append(self.relu)
            
            # c.f. https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
            # if max_pool = False, stride of Conv2d should be set to 2
            if max_pool:
                layers_mean.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))  
        if downsample=="max":
            layers_mean.append(nn.MaxPool2d(kernel_size=5, stride=5, padding=0))
        elif downsample=="average":
            layers_mean.append(nn.AvgPool2d(kernel_size=5, stride=5, padding=0))
        self.conv_layers_mean = nn.Sequential(*layers_mean)
        
        layers_var = list()
        for i in range(len(out_features)):
            in_channels = in_features if i == 0 else out_features[i-1]
            layers_var.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_features[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1))
            layers_var.append(nn.BatchNorm2d(num_features=out_features[i]))
            layers_var.append(self.relu)
            
            # c.f. https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
            # if max_pool = False, stride of Conv2d should be set to 2
            if max_pool:
                layers_var.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))  
        if downsample=="max":
            layers_var.append(nn.MaxPool2d(kernel_size=5, stride=5, padding=0))
        elif downsample=="average":
            layers_var.append(nn.AvgPool2d(kernel_size=5, stride=5, padding=0))
        self.conv_layers_var = nn.Sequential(*layers_var)
        
        self.conv_output_mean = nn.Conv2d(in_channels=out_features[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)
        self.conv_output_var = nn.Conv2d(in_channels=out_features[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        mean = self.conv_layers_mean(x)
        mean = self.conv_output_mean(mean)
        
        var = self.conv_layers_var(x)
        var = self.conv_output_var(var)
        return torch.cat((mean, var), dim=1)

class SimpleFCN_kernel(nn.Module):
    def __init__(self,
                 in_features=4,
                 out_features = (16, 32, 64, 128, 128, 128),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1,
                 max_pool=False):
        """
        A simple fully convolutional neural network.
        Args:
            in_features: input channel dimension (we give images with 4 channels).
            out_features: list of channel feature dimensions.
            num_outputs: number of the output dimension
        """
        super(SimpleFCN_kernel, self).__init__()
        self.relu = nn.ReLU(inplace=True) # inplace=True can sometimes slightly decrease the memory usage
        layers = list()
        for i in range(len(out_features)):
            in_channels = in_features if i == 0 else out_features[i-1]
            if i == len(out_features)//2-1 or i == len(out_features)//2:
                layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_features[i], 
                                    kernel_size=5, stride=stride, padding=2))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, 
                                        out_channels=out_features[i], 
                                        kernel_size=kernel_size, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(num_features=out_features[i]))
            layers.append(self.relu)
            
            # c.f. https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
            # if max_pool = False, stride of Conv2d should be set to 2
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            
        self.conv_layers = nn.Sequential(*layers)
        
        self.conv_output = nn.Conv2d(in_channels=out_features[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        predictions = self.conv_output(x)
        return predictions


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, leaky_relu=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, leaky_relu=leaky_relu)
        self.down1 = Down(64, 128, leaky_relu)
        self.down2 = Down(128, 256, leaky_relu)
        self.down3 = Down(256, 512, leaky_relu)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, leaky_relu)
        self.up1 = Up(1024, 512 // factor, bilinear, leaky_relu)
        self.up2 = Up(512, 256 // factor, bilinear, leaky_relu)
        self.up3 = Up(256, 128 // factor, bilinear, leaky_relu)
        self.up4 = Up(128, 64, bilinear, leaky_relu)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Net(nn.Module):
    def __init__(self, model_name, in_features=4, num_outputs=1, out_features=(16, 32, 64, 128), max_pool=False, downsample=None,
                 bilinear=False, leaky_relu=False):
        super(Net, self).__init__()
        self.model_name = model_name
        if self.model_name == 'fcn_4':
            self.model = SimpleFCN(in_features, out_features, num_outputs, max_pool=max_pool)
        elif self.model_name == 'fcn_6':
            self.model = SimpleFCN(in_features, (16, 32, 64, 128, 128, 128), num_outputs, max_pool=max_pool)
        elif self.model_name == 'fcn_6_gaussian' or self.model_name == 'fcn_6_adf': # splitted network for Gaussian Deep Learning
            self.model = SimpleFCN_Gaussian(in_features, (16, 32, 64, 128, 128, 128), num_outputs=1, max_pool=max_pool, downsample=downsample)
        elif self.model_name == 'fcn_6_kernel':
            self.model = SimpleFCN_kernel(in_features, (16, 32, 64, 128, 128, 128), num_outputs, max_pool=max_pool)
        elif self.model_name == 'unet':
            self.model = UNet(in_features, num_outputs, bilinear, leaky_relu)
        else:
            raise NotImplementedError(f'unknown model name {model_name}')
        
    def forward(self, x):
        return self.model(x)
  
if __name__ == "__main__":
    #model = Net("fcn_4")
    #model = Net("fcn_6_kernel")
    #model = Net("unet", 1, leaky_relu=True)
    model = Net('fcn_6_gaussian', num_outputs=2, in_features=5)
    #model.model.conv_layers_mean.requires_grad_(False)
    #model.model.conv_layers_var.requires_grad_(False)
    #model.model.conv_output_var.requires_grad_(False)
    summary(model, (5, 15, 15), batch_size=256, device="cpu")