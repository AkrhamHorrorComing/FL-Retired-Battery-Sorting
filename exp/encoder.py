import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        一个为OOD检测设计的、更简单的自编码器。
        - 网络结构简化
        - 移除了BatchNorm和Dropout以降低泛化能力
        - 瓶颈层(hidden_dim)应该设置得较小
        """
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),  # 大幅减小隐藏层维度
            nn.ReLU(inplace=True),  # 可以使用简单的ReLU或LeakyReLU
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, hidden_dim)  # 狭窄的瓶颈层
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, input_dim)  # 输出层通常不加激活函数
        )

        # 仍然可以使用好的权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded