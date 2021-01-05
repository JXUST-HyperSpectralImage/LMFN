import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary


# class SpatialBlock(nn.Sequential):
#     def __init__(self):
#         super(SpatialBlock, self).__init__()
#         self.spatial_conv_layer = nn.Sequential(
#             nn.Conv3d(1, 24, kernel_size=(24, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)), nn.BatchNorm3d(24))


# class _DenseLayer(nn.Sequential):
#     def __init__(self, input_channels, drop_rate=0):
#         super(_DenseLayer, self).__init__()
#         self.add_module('norm1', nn.BatchNorm3d(input_channels))
#         self.add_module('relu1', nn.ReLU(inplace=True))
#         self.add_module('conv3d', nn.Conv3d(input_channels, input_channels, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 1, 1)))
#         self.drop_rate = drop_rate
#
#     def forward(self, inputs):
#         new_features = super(_DenseLayer, self).forward(inputs)
#         if self.drop_rate > 0:
#             new_features = F.dropout3d(new_features, p=self.drop_rate, training=self.training)
#         print('inputs', inputs.shape)
#         print('new_features', new_features.shape)
#         return torch.cat([inputs, new_features], 1)
#
#
# class _DenseBlock(nn.Sequential):
#     def __init__(self, num_layers, input_channels, growth_rate, drop_rate):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             print('dense block features', input_channels + growth_rate * i)
#             layer = _DenseLayer(input_channels + growth_rate * i, drop_rate)
#             self.add_module(f'DenseLayer{i + 1}', layer)


class MCSSN(nn.Module):

    def __init__(self, in_channel, classes, kernel_nums=24, spe_kernel_depth=7, drop_rate=0):
        super(MCSSN, self).__init__()
        # Spectral Featrue Learning
        self.spectral_conv1 = nn.Sequential(
            nn.Conv3d(1, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)), nn.BatchNorm3d(kernel_nums))
        # spe_feature_pad = self.feature_pad(spe_kernel_depth)

        self.spectral_residual_block = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0)),
            nn.BatchNorm3d(kernel_nums),
            nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0)),
            nn.BatchNorm3d(kernel_nums)
        )
        # spectral transform to spatial
        spa_kernel_depth = self.spa_feature_depth(kernel_depth=spe_kernel_depth, in_channel=in_channel)
        self.spe_to_spa = nn.Sequential(
            nn.Conv3d(kernel_nums, kernel_nums//3, kernel_size=(spe_kernel_depth, 1, 1), stride=(1, 1, 1), padding=(spe_kernel_depth//2, 0, 0)),
            nn.BatchNorm3d(kernel_nums//3)
        )

        spa_kernels = kernel_nums//3
        self.spatial_conv1 = nn.Conv3d(spa_kernels, spa_kernels, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 1, 1))
        self.spatial_conv2 = nn.Conv3d(spa_kernels*2, spa_kernels*2, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 1, 1))

        # self.spatial_dense_layer = _DenseBlock(num_layers=4, input_channels=kernel_nums//4, growth_rate=6, drop_rate=drop_rate)

        self.end_conv = nn.Conv3d(kernel_nums, kernel_nums, kernel_size=(spa_kernel_depth, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.pool = nn.AdaptiveAvgPool3d((kernel_nums, 1, 1))
        self.fc = nn.Linear(kernel_nums, classes)

        self.drop_rate = drop_rate

    # @staticmethod
    # def feature_pad(kernel_depth):
    #     depth_pad = kernel_depth // 2
    #     return depth_pad

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
        x0 = self.spectral_conv1(x)
        if self.drop_rate > 0:
            F.dropout3d(x0, p=0.5)

        # frist spectral residual block
        x_res = self.spectral_residual_block(x0)
        if self.drop_rate > 0:
            F.dropout3d(x_res, p=0.5)

        x1 = x0 + x_res

        # spectral tranform to spatial
        x2 = self.spe_to_spa(x1)

        if self.drop_rate > 0:
            F.dropout3d(x2, p=0.5)
        # x2 = self.dim_trans(x2)
        # print(x2.shape)

        x0 = self.spe_to_spa(x0)


        x2 = x2 + x0

        # spatial initial conv
        # x3 = self.spatial_dense_layer(x2)
        x3_1 = self.spatial_conv1(x2)
        x3_1 = torch.cat([x3_1, x2], 1)

        x3_2 = self.spatial_conv2(x3_1)
        x3 = torch.cat([x3_2, x2], 1)

        x3 = x3 + x1
        x = self.end_conv(x3)
        x = self.dim_trans(x)

        x = self.pool(x)
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
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0003)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'PaviaU':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0003)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'KSC':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.2)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 16)
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
        kwargs.setdefault('kernel_nums', 24)

    model = MCSSN(
        in_channel=kwargs['n_bands'],
        classes=kwargs['n_classes'],
        kernel_nums=kwargs['kernel_nums'])
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(kwargs['device'])
    optimizer = optim.SGD(
        model.parameters(),
        lr=kwargs['lr'],
        weight_decay=0.0005,
        momentum=0.9)
    kwargs.setdefault(
        'scheduler',
        optim.lr_scheduler.StepLR(
            optimizer,
            step_size=33333,
            gamma=0.1))
    kwargs.setdefault('supervision', 'full')
    # 使用中心像素点作为监督信息
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model = MCSSN(in_channel=200, classes=17, kernel_nums=24, spe_kernel_depth=7, drop_rate=0).to(device)
    with torch.no_grad():
        summary(model, (1, 200, 7, 7))
