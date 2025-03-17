from  dataloader import *


class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ConvBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x
    

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel//2, 2, stride=2)
        self.conv = ConvBlock(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
        


class AttentionBlock(nn.Module):
    def __init__(self,fg,fl,n_coefficients):
        super(AttentionBlock,self).__init__()

        self.wgate = nn.Sequential(
            nn.Conv2d(fg,n_coefficients,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.wx = nn.Sequential(
            nn.Conv2d(fl,n_coefficients,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients,1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,gate,skip):
        g1 = self.wgate(gate)
        x1 = self.wx(skip)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = skip*psi 
        return out 



class AttentionUnet(nn.Module):
    def __init__(self, img_ch=3,out_ch=1):
        super(AttentionUnet,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = ConvBlock(in_channel=img_ch,out_channel=64)
        self.conv2 = ConvBlock(in_channel=64,out_channel=128)
        self.conv3 = ConvBlock(in_channel=128,out_channel=256)
        self.conv4 = ConvBlock(in_channel=256,out_channel=512)
        self.conv5 = ConvBlock(in_channel=512,out_channel=1024)

        self.up5 = Up(in_channel=1024,out_channel=512,bilinear=False)
        self.Att5 = AttentionBlock(fg=512, fl=512, n_coefficients=256)
        self.upconv5 = ConvBlock(in_channel=1024,out_channel=512)

        self.Up4 = Up(512, 256)
        self.Att4 = AttentionBlock(fg=256, fl=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)


        self.Up3 = Up(256, 128)
        self.Att3 = AttentionBlock(fg=128, fl=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)


        self.Up2 = Up(128, 64)
        self.Att2 = AttentionBlock(fg=64, fl=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.Maxpool(enc1)
        enc2 = self.conv2(enc2)
        enc3 = self.Maxpool(enc2)
        enc3 = self.conv3(enc3)
        enc4 = self.Maxpool(enc3)
        enc4 = self.conv4(enc4)
        enc5 = self.Maxpool(enc4)
        enc5 = self.conv5(enc5)

       
        dec5 = self.up5(enc5, enc4)
        at5 = self.Att5(dec5, enc4)
        dec5 = torch.cat((at5, dec5), dim=1)
        dec5 = self.upconv5(dec5)  

        
        dec4 = self.Up4(dec5, enc3)   
        at4 = self.Att4(dec4, enc3)
        dec4 = torch.cat((at4, dec4), dim=1)
        dec4 = self.UpConv4(dec4)

        dec3 = self.Up3(dec4, enc2)
        at3 = self.Att3(dec3, enc2)
        dec3 = torch.cat((at3, dec3), dim=1)
        dec3 = self.UpConv3(dec3)

        dec2 = self.Up2(dec3, enc1)  
        at2 = self.Att2(dec2, enc1)
        dec2 = torch.cat((at2, dec2), dim=1)
        dec2 = self.UpConv2(dec2)

        out = self.Conv(dec2)  

        return out


    