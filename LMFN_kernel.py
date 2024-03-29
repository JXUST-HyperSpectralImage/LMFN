import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary
import numpy as np
from ptflops import get_model_complexity_info


class LMFN(nn.Module):
#    @staticmethod
#    def weight_init(m):
#        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#            torch.nn.init.kaiming_normal_(m.weight)
#            torch.nn.init.zeros_(m.bias)

    def __init__(self, in_channel, classes, kernel_nums=1, spe_kernel_depth=7, spa_kernel_size=5, init_conv_stride=1,
                 drop_rate=0.5, bias=True):
        super(LMFN, self).__init__()
        # Spectral Featrue Learning
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(init_conv_stride, 1, 1),
                      padding=(spe_kernel_depth // 2, 0, 0), bias=bias),
            nn.BatchNorm3d(kernel_nums))

        self.spectral_conv1 = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1),
                      padding=(spe_kernel_depth // 2, 0, 0), bias=bias),
            nn.BatchNorm3d(kernel_nums))
        self.spectral_conv2 = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1),
                      padding=(spe_kernel_depth // 2, 0, 0), bias=bias),
            nn.BatchNorm3d(kernel_nums))
        self.spectral_conv3 = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1),
                      padding=(spe_kernel_depth // 2, 0, 0), bias=bias),
            nn.BatchNorm3d(kernel_nums))
        self.spectral_conv4 = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1),
                      padding=(spe_kernel_depth // 2, 0, 0), bias=bias),
            nn.BatchNorm3d(kernel_nums))
        
        self.transformation_layer = nn.Conv3d(kernel_nums, 1, kernel_size=(1, 1, 1))
        # spectral transform to spatial
        spa_kernel_depth = self.spa_feature_depth(kernel_depth=spe_kernel_depth, in_channel=in_channel,
                                                  padding=spe_kernel_depth // 2, stride=init_conv_stride)

        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=spa_kernel_size, padding=5 // 2,
                      groups=spa_kernel_depth),
            nn.BatchNorm2d(spa_kernel_depth))
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=spa_kernel_size, padding=5 // 2,
                      groups=spa_kernel_depth),
            nn.BatchNorm2d(spa_kernel_depth))
        self.spatial_conv3 = nn.Sequential(
            nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=spa_kernel_size, padding=5 // 2,
                      groups=spa_kernel_depth),
            nn.BatchNorm2d(spa_kernel_depth))

        self.spatial_end_conv1 = nn.Sequential(
            nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=5, padding=5 // 2, groups=spa_kernel_depth))
        self.spatial_end_conv2 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=3, padding=1,
                                           groups=spa_kernel_depth)
        self.spatial_end_conv3 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=1, padding=0,
                                           groups=spa_kernel_depth)

        self.pool = nn.AdaptiveAvgPool3d((spa_kernel_depth, 1, 1))
        self.fc = nn.Linear(spa_kernel_depth, classes)

        self.drop_rate = drop_rate
#        self.apply(self.weight_init)
    @staticmethod
    def spa_feature_depth(kernel_depth, in_channel, padding, stride, dilation=1):
        depth = (in_channel + 2 * padding - dilation * (kernel_depth - 1) - 1) // stride + 1
        return depth

    @staticmethod
    def adjacency_matrix(data):
        x = data.shape[-2] // 2
        y = data.shape[-1] // 2
        adjacency_matrix = torch.Tensor().to(torch.device('cuda'))
        try:
            center_pixel = data[:, :, :, x:x+1, y:y+1]
            center_pixel = center_pixel.view(data.shape[0], data.shape[1], data.shape[2], 1, 1).repeat(1, 1, 1, data.shape[-2],
                                                                                        data.shape[-1])
            adjacency_matrix = center_pixel.mul(data)
            adjacency_matrix = torch.sum(center_pixel.mul(data), dim=1)
            adjacency_matrix = torch.unsqueeze(adjacency_matrix, 1)
            adjacency_matrix = adjacency_matrix.repeat(1, data.shape[1], 1, 1, 1)
        except:
            center_pixel = data[:, x, y]
            center_pixel = center_pixel.view(data.shape[0], 1, 1).repeat(1, data.shape[-2], data.shape[-1])
            adjacency_matrix = center_pixel.mul(data)
            adjacency_matrix = torch.sum(center_pixel.mul(data), dim=0)
            adjacency_matrix = torch.unsqueeze(adjacency_matrix, 0)
            adjacency_matrix = adjacency_matrix.repeat(data.shape[0], 1, 1)
        adjacency_matrix = F.sigmoid(adjacency_matrix)
        return adjacency_matrix

    def forward(self, x):
        # spectral initial conv
        x0 = self.init_conv(x)

        spectral_mask0 = self.adjacency_matrix(torch.squeeze(x0))  # frist mask

        if self.drop_rate > 0:
            F.dropout3d(x0, p=self.drop_rate)

        # spectral residual block
        x1 = self.spectral_conv1(x0)
        if self.drop_rate > 0:
            F.dropout3d(x1, p=self.drop_rate)

        x2 = self.spectral_conv2(x1)
        if self.drop_rate > 0:
            F.dropout3d(x2, p=self.drop_rate)

        # frist residual
        x2 = x0 + x2

        spectral_mask1 = self.adjacency_matrix(torch.squeeze(x2))  # second mask

        x3 = self.spectral_conv3(x2)
        if self.drop_rate > 0:
            F.dropout3d(x3, p=self.drop_rate)

        # spectral tranform to spatial
        x4 = self.spectral_conv4(x3)
        if self.drop_rate > 0:
            F.dropout3d(x4, p=self.drop_rate)

        # second residual
        x4 = x2 + x4

        spectral_mask2 = self.adjacency_matrix(x4)  # thrid mask

        x4 = self.transformation_layer(x4)
        x4 = torch.squeeze(x4)
        # TODO:Please release this line when calculate FLOPs
        # x4 = x4.reshape(1, x4.size(0), x4.size(-2), x4.size(-1))
        # spatial depthwise conv
        x5 = self.spatial_conv1(x4)
        
        x5 = torch.squeeze(self.transformation_layer(x0.mul(spectral_mask0))) + x5  # frist attention

        x6 = self.spatial_conv2(x5)
        
        x6 = torch.squeeze(self.transformation_layer(x2.mul(spectral_mask1))) + x6  # second attention

        x7 = self.spatial_conv3(x6)

        x7 = torch.squeeze(self.transformation_layer(torch.unsqueeze(x4, 1).mul(spectral_mask2))) + x7  # thrid attention

        # multi-scale feature fusion

        x8 = self.spatial_end_conv1(x7)  # 5x5 conv
        x8 = F.gelu(x8)

        x9 = self.spatial_end_conv2(x8)  # 3x3 conv
        x9 = F.gelu(x9)

        x10 = self.spatial_end_conv3(x9)  # 1x1 conv

        x10 = F.gelu(x10)

        x = x8 + x9 + x10

        x = F.gelu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters
    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    kwargs.setdefault('device', torch.device('cpu'))
    weights = torch.ones(kwargs['n_classes'])
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(kwargs['device'])
    kwargs.setdefault('weights', weights)
    kwargs.setdefault('epoch', 100)
    kwargs.setdefault('patch_size', 9)
    kwargs.setdefault('lr', 0.01)
    kwargs.setdefault('batch_size', 32)
    kwargs.setdefault('kernel_depth', 7)

    model = LMFN(
        in_channel=kwargs['n_bands'],
        classes=kwargs['n_classes'],
        kernel_nums=kwargs['kernel_nums'],
        spe_kernel_depth=kwargs['kernel_depth'],
        init_conv_stride=2,
        drop_rate=0.5)
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(kwargs['device'])
    optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                        patience=kwargs['epoch'] // 10, verbose=True))
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_size = 9
    # IN
    bands = 200
    classes = 17
    # UP
#    bands = 103
#    classes = 10
    # KSC
    bands = 176
    classes = 14
    
    model = LMFN(in_channel=bands, classes=classes, kernel_nums=1, spe_kernel_depth=7, init_conv_stride=2, drop_rate=0.5).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: {}==>{:.2f}M".format(total, total/1e6))
    flops, params = get_model_complexity_info(model, (1, bands, patch_size, patch_size), as_strings=False, print_per_layer_stat=True, verbose=True)
    print("Flops:{}M ==> {}G".format(flops/1e6, flops/1e9))
    print("Params:{}M".format(params/1e6))
#    with torch.no_grad():
#        summary(model, (1, bands, patch_size, patch_size))

