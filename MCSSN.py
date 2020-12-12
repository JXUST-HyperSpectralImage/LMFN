import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary


class SpatialBlock(nn.Sequential):
    def __init__(self):
        super(SpatialBlock, self).__init__()
        self.spatial_conv_layer = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(24, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(24))


class ZhongEtAl(nn.Module):

    def __init__(self, in_channel, classes, kernel_nums=24, spe_kernel_depth=7, drop_out=False):
        super(ZhongEtAl, self).__init__()
        self.drop_out = drop_out
        # Spectral Featrue Learning
        self.spectral_conv1 = nn.Sequential(
            nn.Conv3d(1, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(kernel_nums)
            )
        spe_feature_pad = self.feature_pad(spe_kernel_depth)

        self.spectral_block = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1),
                      padding=(spe_feature_pad, 0, 0)),
            nn.BatchNorm3d(kernel_nums),
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1),
                      padding=(spe_feature_pad, 0, 0)),
            nn.BatchNorm3d(kernel_nums)
            )
        # spectral transform to spatial
        spa_kernel_depth = self.spa_feature_depth(kernel_depth=spe_kernel_depth, in_channel=in_channel)
        print(spa_kernel_depth)
        self.spe_to_spa = nn.Sequential(
            nn.Conv3d(kernel_nums, spa_kernel_depth, kernel_size=(spa_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(spa_kernel_depth)
            )
        # Spatial Featrue Learning
        self.spatial_conv1 = nn.Sequential(
            nn.Conv3d(1, spa_kernel_depth, kernel_size=(spa_kernel_depth, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(spa_kernel_depth)
            )

        self.spatial_conv_layer = nn.Sequential(
            nn.Conv3d(1, spa_kernel_depth, kernel_size=(spa_kernel_depth, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(spa_kernel_depth))

        self.pool = nn.AdaptiveAvgPool3d((spa_kernel_depth, 1, 1))
        self.fc = nn.Linear(spa_kernel_depth, classes)

    @staticmethod
    def feature_pad(kernel_depth):
        depth_pad = kernel_depth // 2
        return depth_pad

    @staticmethod
    def spa_feature_depth(kernel_depth, in_channel, padding=0, dilation=1, stride=2):
        depth = (in_channel + 2 * padding - dilation * (kernel_depth - 1) - 1) // stride + 1
        return depth

    @staticmethod
    def dim_trans(x):
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 1)
        return x

    def forward(self, x):
        # spectral initial conv
        x = self.spectral_conv1(x)
        if self.drop_out:
            F.dropout3d(x, p=0.5)

        # frist spectral residual block
        x_res = self.spectral_block(x)
        if self.drop_out:
            F.dropout3d(x_res, p=0.5)

        x1 = x + x_res

        # second spectral residual block
        x_res = self.spectral_block(x1)
        if self.drop_out:
            F.dropout3d(x_res, p=0.5)

        x2 = x1 + x_res

        # spectral tranform to spatial
        x = self.spe_to_spa(x2)
        if self.drop_out:
            F.dropout3d(x, p=0.5)
        # x = self.dim_trans(x)

        # spatial initial conv
        x = self.dim_trans(x)
        x = self.spatial_conv1(x)
        if self.drop_out:
            F.dropout3d(x, p=0.5)
        x = self.dim_trans(x)

        # frist spatial residual block
        x_res = self.spatial_conv_layer(x)
        if self.drop_out:
            F.dropout3d(x_res, p=0.5)
        x_res = self.dim_trans(x_res)

        x_res = self.spatial_conv_layer(x_res)
        if self.drop_out:
            F.dropout3d(x_res, p=0.5)
        x_res = self.dim_trans(x_res)

        x1 = self.spe_to_spa(x1)
        x1 = self.dim_trans(x1)
        x3 = x + x_res + x1


        # second spatial residual block
        x_res = self.spatial_conv_layer(x3)
        if self.drop_out:
            F.dropout3d(x, p=0.5)
        x_res = self.dim_trans(x_res)
        x_res = self.spatial_conv_layer(x_res)
        if self.drop_out:
            F.dropout3d(x_res, p=0.5)
        x_res = self.dim_trans(x_res)

        x2 = self.spe_to_spa(x2)
        x2 = self.dim_trans(x2)

        x4 = x3 + x_res + x2
        # pooling layer
        x = self.pool(x4)
        x = x.view(x.size(0), -1)

        # fully connected layer
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
        kwargs.setdefault('training_percentage', 0.2)
        kwargs.setdefault('validation_group', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0003)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'PaviaU':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_group', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0003)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'KSC':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.2)
        kwargs.setdefault('validation_group', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 16)
    elif name == 'Botswana':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_group', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'HoustonU':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_group', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'Salinas':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_group', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)

    model = ZhongEtAl(in_channel=kwargs['n_bands'], classes=kwargs['n_classes'], kernel_nums=kwargs['kernel_nums'])
    optimizer = optim.RMSprop(model.parameters(), lr=kwargs['lr'])
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(kwargs['device'])
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                        patience=kwargs['epoch'] // 4, verbose=True))
    kwargs.setdefault('3D_data', True)
    kwargs.setdefault('supervision', 'full')
    # 数据增强默认关闭
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    # 使用中心像素点作为监督信息
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model = ZhongEtAl(in_channel=200, classes=17, kernel_nums=24, spe_kernel_depth=7, drop_out=True).to(device)
    with torch.no_grad():
        summary(model, (1, 200, 7, 7))
