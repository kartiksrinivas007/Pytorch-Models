import torch
import torch.nn as nn
import torchvision.transforms.functional as tf



class doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self , x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels=1, features =[64, 128, 256, 512]):
        super(UNET,self).__init__()
        self.downs = nn.ModuleList() # we need to do module.evla thats why
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        
        for feature in features:
            self.downs.append(doubleconv(in_channels, feature))
            in_channels = feature # pretty cool , just moving forward to add more layers
            
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride = 2,
                )
            )
            self.ups.append(doubleconv(feature*2, feature))
        
        self.bottleneck = (doubleconv(features[-1], features[-1]*2))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1] # reverse, since I need the last value of x as the first to the next one 
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2] # steps of two # remeber that these are all references to objects and not copies
            if (x.shape != skip_connection.shape): # maybe cause of the maxpool 2d rounding it off, since kernel_size is 2, the number of pixels should be div by 16
                x = tf.resize(x, size = skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim = 1) # along channel = () batch, channle, height , width)
            x = self.ups[idx + 1](concat_skip)
            # how does the skip connection work, cat
        return self.final_conv(x)
    
