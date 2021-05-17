import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary
import numpy as np


class MI3DCNN(nn.Module):

    def __init__(self, in_channel, classes, kernel_nums=1, spe_kernel_depth=7, init_conv_stride=1, drop_rate=0.5, bias=True):
        super(MI3DCNN, self).__init__()
        # Spectral Featrue Learning
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(init_conv_stride, 1, 1), padding=(spe_kernel_depth//2, 0, 0), bias=bias), nn.BatchNorm3d(kernel_nums))
        
        self.spectral_conv1 = nn.Sequential(
        nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0), bias=bias),
        nn.BatchNorm3d(kernel_nums))
        self.spectral_conv2 = nn.Sequential(
        nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0), bias=bias),
        nn.BatchNorm3d(kernel_nums))
        self.spectral_conv3 = nn.Sequential(
        nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0), bias=bias),
        nn.BatchNorm3d(kernel_nums))
        
        self.spectral_end_conv = nn.Sequential(
        nn.Conv3d(kernel_nums, 1, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0), bias=bias),
        nn.BatchNorm3d(1))

        # spectral transform to spatial
        spa_kernel_depth = self.spa_feature_depth(kernel_depth=spe_kernel_depth, in_channel=in_channel, padding=spe_kernel_depth//2, stride=init_conv_stride)
        print('spa_kernel_depth:', spa_kernel_depth)
        self.spatial_conv1 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=5, padding=5//2, groups=spa_kernel_depth)
        self.spatial_conv2 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=5, padding=5//2, groups=spa_kernel_depth)
        self.spatial_conv3 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=5, padding=5//2, groups=spa_kernel_depth)
        
        self.adaptive_transform = nn.Conv3d(kernel_nums, spa_kernel_depth, kernel_size=(spa_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        
        self.spatial_end_conv1 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=3, padding=1, groups=spa_kernel_depth)
        self.spatial_end_conv2 = nn.Conv2d(spa_kernel_depth, spa_kernel_depth, kernel_size=1, padding=0, groups=spa_kernel_depth)
        
        self.pool = nn.AdaptiveAvgPool3d((spa_kernel_depth, 1, 1))
        self.fc = nn.Linear(spa_kernel_depth, classes)

        self.drop_rate = drop_rate

    @staticmethod
    def spa_feature_depth(kernel_depth, in_channel, padding, stride, dilation=1):
        depth = (in_channel + 2 * padding - dilation * (kernel_depth - 1) - 1) // stride + 1
        return depth

    @staticmethod
    def adjacency_matrix(data):
        x = data.shape[-2]//2
        y = data.shape[-1]//2
        adjacency_matrix = torch.Tensor().to(torch.device('cuda'))
        try:
            center_pixel = data[:, :, x, y]
            center_pixel = center_pixel.view(data.shape[0], data.shape[1], 1, 1).repeat(1, 1, data.shape[-2], data.shape[-1])
#            data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
            adjacency_matrix = center_pixel.mul(data)
            adjacency_matrix = torch.sum(center_pixel.mul(data), dim=1)
            adjacency_matrix = torch.unsqueeze(adjacency_matrix, 1)
            adjacency_matrix = adjacency_matrix.repeat(1, data.shape[1], 1, 1)
        except:
            center_pixel = data[:, x, y]
            center_pixel = center_pixel.view(data.shape[0], 1, 1).repeat(1, data.shape[-2], data.shape[-1])
            adjacency_matrix = center_pixel.mul(data)
            adjacency_matrix = torch.sum(center_pixel.mul(data), dim=0)
            adjacency_matrix = torch.unsqueeze(adjacency_matrix, 0)
            adjacency_matrix = adjacency_matrix.repeat(data.shape[0], 1, 1)
        adjacency_matrix = torch.sigmoid(adjacency_matrix)
        return adjacency_matrix


    def forward(self, x):
        # spectral initial conv
        x0 = self.init_conv(x)

#        spectral_mask0 = self.adjacency_matrix(torch.squeeze(self.adaptive_transform(x0)))
        spectral_mask0 = self.adjacency_matrix(torch.squeeze(x0))
        
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
        spectral_mask1 = self.adjacency_matrix(torch.squeeze(x2))
        
        x3 = self.spectral_conv3(x2)
        if self.drop_rate > 0:
            F.dropout3d(x3, p=self.drop_rate)
    
        # Residual connection
#        x3 = x0 + x3
        
#        spectral_mask1 = self.adjacency_matrix(torch.squeeze(self.adaptive_transform(x3)))
        # spectral tranform to spatial
        x4 = self.spectral_end_conv(x3)
        if self.drop_rate > 0:
            F.dropout3d(x4, p=self.drop_rate)
            
        # second residual
        x4 = x2 + x4
        x4 = torch.squeeze(x4)
        
        spectral_mask2 = self.adjacency_matrix(x4)
#        x4 = x4.view(x4.size(1), 1, x4.size(2), x4.size(3))
        # spatial depthwise conv
        x5 = self.spatial_conv1(x4)
        x5 = torch.squeeze(x0).mul(spectral_mask0) + x5
#        x5 = torch.squeeze(self.adaptive_transform(x0)).mul(spectral_mask0) + x5
        
        
        x6 = self.spatial_conv2(x5)
        x6 = torch.squeeze(x2).mul(spectral_mask1) + x6
        
        x7 = self.spatial_conv3(x6)
        x7 = x4.mul(spectral_mask2) + x7
        
#        x7 = torch.squeeze(self.adaptive_transform(x3)).mul(spectral_mask1) + x7
#        x7 = torch.squeeze(x3).mul(spectral_mask1) + x7
        
#        x7 = x5 + x7
        x7 = F.relu(x7) # 5x5 conv
        
        x8 = self.spatial_end_conv1(x7) # 3x3 conv
        
        x8 = F.relu(x8)
        
        x9 = self.spatial_end_conv2(x8) # 1x1 conv
                
        x9 = F.relu(x9)
        
        x = x7 + x8 + x9
        
#        x10 = self.spatial_end_conv(x9)

        x = F.relu(x)

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
    kwargs.setdefault('batch_size', 16)
    kwargs.setdefault('patch_size', 7)
    kwargs.setdefault('epoch', 200)

    if name == 'IndianPines':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0003)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 1)
    elif name == 'PaviaU':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.02)
        kwargs.setdefault('validation_percentage', 0.02)
        # learning rate
        kwargs.setdefault('lr', 0.01)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 1)
    elif name == 'KSC':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.03)
        kwargs.setdefault('validation_percentage', 0.03)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 1)
    elif name == 'Botswana':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'HoustonU':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'Salinas':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 8)

    model = MI3DCNN(
        in_channel=kwargs['n_bands'],
        classes=kwargs['n_classes'],
        kernel_nums=kwargs['kernel_nums'],
        spe_kernel_depth=kwargs['kernel_depth'],
        init_conv_stride=1,
        drop_rate=0)
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(kwargs['device'])
#    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
#    optimizer = optim.RMSprop(model.parameters(), lr=kwargs['lr'])
    optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001, momentum=0.9)
        
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                        patience=kwargs['epoch']//10, verbose=True))
#    kwargs.setdefault(
#        'scheduler',
#        optim.lr_scheduler.StepLR(
#            optimizer,
#            step_size=33333,
#            gamma=0.1, verbose=True))
    kwargs.setdefault('supervision', 'full')
    # 使用中心像素点作为监督信息
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model = MI3DCNN(in_channel=200, classes=17, kernel_nums=24, spe_kernel_depth=7, drop_rate=0).to(device)
    with torch.no_grad():
        summary(model, (1, 200, 7, 7))
