import torch
from torch import nn


class ResUnet3D(nn.Module):
    def __init__(self, ms_flag, mtp_flag, mts_flag, train_flag, drop_flag, drop_rate=0, g_size=8, init_flt_size=32):
        super().__init__()
        self.fil_size = init_flt_size
        self.DropoutFlag = drop_flag
        self.MultiScale = ms_flag
        self.MultiTaskPar = mtp_flag
        self.MultiTaskSeq = mts_flag
        self.IsTraining = train_flag
        self.DropRate = drop_rate
        self.GroupSize = g_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define layers and parameters here
        self.short_1 = self.conv3_gn(in_channels=1, out_channels=self.fil_size, kernel_size=1, stride=1)

        self.out_1 = self.conv3(in_channels=self.fil_size, out_channels=self.fil_size*2, kernel_size=1, stride=2)

        self.out_2 = self.conv3(in_channels=self.fil_size*2, out_channels=self.fil_size*4, kernel_size=1, stride=2)

        self.out_3 = self.conv3(in_channels=self.fil_size*4, out_channels=self.fil_size*8, kernel_size=1, stride=2)

        # self.out_4 = self.conv3(in_channels=256, out_channels=512, kernel_size=1, stride=2)

        # self.out_5 = self.conv3(in_channels=512, out_channels=1024, kernel_size=1, stride=2)
        #
        # self.out_8 = self.up_sample(in_channels=1024, out_channels=512)
        #
        # self.out_11 = self.up_sample(in_channels=512, out_channels=256)

        self.out_14 = self.up_sample(in_channels=self.fil_size*8, out_channels=self.fil_size*4)

        self.out_17 = self.up_sample(in_channels=self.fil_size*4, out_channels=self.fil_size*2)

        self.out_20 = self.up_sample(in_channels=self.fil_size*2, out_channels=self.fil_size)

        self.color_logit = self.conv3_gn(in_channels=self.fil_size, out_channels=2, kernel_size=1, stride=1)

        self.out = nn.Conv3d(in_channels=self.fil_size, out_channels=2, kernel_size=1, stride=1)

    def conv3(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, device=self.device),
            nn.ReLU(inplace=True)
        )

    # @profile
    def conv3_gn(self, in_channels, out_channels, kernel_size, stride=1, act='relu'):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, device=self.device),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Sigmoid()
        )

    # @profile
    def res_block(self, in_data, in_channels, kernel_size, stride, dilation):
        input_ = in_data
        out_data = torch.zeros_like(in_data)
        for i in range(len(dilation)):
            in_data = input_
            in_data = nn.GroupNorm(self.GroupSize, in_channels, device=self.device)(in_data)
            in_data = nn.ReLU(inplace=True)(in_data)
            in_data = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding=dilation[i],
                                dilation=dilation[i], device=self.device)(in_data)
            in_data = nn.GroupNorm(self.GroupSize, in_channels, device=self.device)(in_data)
            in_data = nn.ReLU(inplace=True)(in_data)
            in_data = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding=dilation[i],
                                dilation=dilation[i], device=self.device)(in_data)
            out_data += in_data
        out_data += input_
        return out_data

    # @profile
    def psp(self, in_data, in_channels):
        gs = self.GroupSize
        output = in_data
        ch_num = int(in_channels / 4)
        if ch_num < self.GroupSize:
            gs = ch_num

        for i in range(4):
            dx = in_data[:, i * ch_num:(i + 1) * ch_num, :, :, :]
            dx = nn.MaxPool3d(2 ** i)(dx)
            dx = nn.Upsample(scale_factor=2 ** i, mode='trilinear')(dx)
            dx = nn.Conv3d(ch_num, ch_num, 1, 1, padding='same', device=self.device)(dx)
            dx = nn.GroupNorm(gs, ch_num, device=self.device)(dx)
            dx = nn.ReLU()(dx)
            output = torch.cat((output, dx), 1)

        output = nn.Conv3d(in_channels*2, in_channels, 1, 1, padding='same', device=self.device)(output)
        output = nn.GroupNorm(self.GroupSize, in_channels, device=self.device)(output)
        output = nn.ReLU(in_channels)(output)
        return output

    # @profile
    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, out_channels, 1, 1, padding='same', device=self.device),
            nn.GroupNorm(self.GroupSize, out_channels, device=self.device),
            nn.ReLU(inplace=True)
        )

    # @profile
    def combine(self, x1, x2, in_channels, out_channels):
        in_data = torch.cat((x1, x2), 1)
        in_data = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding='same',
                            device=self.device)(in_data)
        in_data = nn.GroupNorm(self.GroupSize, out_channels, device=self.device)(in_data)
        in_data = nn.ReLU()(in_data)
        return in_data

    # @profile
    def drop(self, in_data, drop_rate):
        train_flag = self.IsTraining
        in_data = f.dropout(in_data, drop_rate, train_flag)
        return in_data

    # @profile
    def mapp(self, in_data, in_channels, scale_factor):
        in_data = nn.Conv3d(in_channels, 2, 1, 1, device=self.device)(in_data)
        in_data = nn.Upsample(scale_factor=scale_factor, mode='trilinear')(in_data)
        in_data = nn.Sigmoid()(in_data)
        return in_data

    # @profile
    def forward(self, in_data):
        short_1 = self.short_1(in_data)
        short_2 = self.res_block(in_data=short_1, in_channels=self.fil_size, kernel_size=3, stride=1,
                                 dilation=[1, 3, 15, 31])
        in_data = self.out_1(short_2)

        short_3 = self.res_block(in_data=in_data, in_channels=self.fil_size*2, kernel_size=3, stride=1,
                                 dilation=[1, 3, 15])
        in_data = self.out_2(short_3)

        short_4 = self.res_block(in_data=in_data, in_channels=self.fil_size*4, kernel_size=3, stride=1,
                                 dilation=[1, 3, 7])
        in_data = self.out_3(short_4)

        in_data = self.res_block(in_data=in_data, in_channels=self.fil_size*8, kernel_size=3, stride=1,
                                 dilation=[1, 3])
        # in_data = self.out_4(short_5)

        # short_6 = self.res_block(in_data=in_data, in_channels=512, kernel_size=3, stride=1, dilation=[1])
        # in_data = self.out_5(short_6)

        # in_data = self.res_block(in_data=in_data, in_channels=1024, kernel_size=3, stride=1, dilation=[1])

        in_data = self.psp(in_data, in_channels=self.fil_size*8)

        # in_data = self.out_8(in_data)
        # in_data = self.combine(short_6, in_data, in_channels=512)
        # in_data = self.res_block(in_data, in_channels=512,  kernel_size=3, stride=1, dilation=[1])
        # if self.MultiScale:
        #     output1 = self.mapp(in_data, 512, 32)
        # if self.DropoutFlag:
        #     in_data = self.drop(in_data, self.DropRate)
        #
        # in_data = self.out_11(in_data)
        # in_data = self.combine(short_5, in_data, in_channels=256)
        # in_data = self.res_block(in_data=in_data, in_channels=256, kernel_size=3, stride=1, dilation=[1, 3, 15])
        # if self.MultiScale:
        #     output2 = self.mapp(in_data, 256, 16)
        # if self.DropoutFlag:
        #     in_data = self.drop(in_data, self.DropRate)

        in_data = self.out_14(in_data)
        in_data = self.combine(short_4, in_data, in_channels=self.fil_size*8, out_channels=self.fil_size*4)
        in_data = self.res_block(in_data=in_data, in_channels=self.fil_size*4, kernel_size=3, stride=1,
                                 dilation=[1, 3, 7])
        if self.MultiScale:
            output3 = self.mapp(in_data, self.fil_size*4, 4)
        if self.DropoutFlag:
            in_data = self.drop(in_data, self.DropRate)

        in_data = self.out_17(in_data)
        in_data = self.combine(short_3, in_data, in_channels=self.fil_size*4, out_channels=self.fil_size*2)
        in_data = self.res_block(in_data=in_data, in_channels=self.fil_size*2, kernel_size=3, stride=1,
                                 dilation=[1, 3, 15])
        if self.MultiScale:
            output4 = self.mapp(in_data, self.fil_size*2, 2)
        if self.DropoutFlag:
            in_data = self.drop(in_data, self.DropRate)

        in_data = self.out_20(in_data)
        in_data = self.combine(short_2, in_data, in_channels=self.fil_size*2, out_channels=self.fil_size)
        in_data = self.res_block(in_data=in_data, in_channels=self.fil_size, kernel_size=3, stride=1,
                                 dilation=[1, 3, 15, 31])
        if self.MultiScale:
            output5 = self.mapp(in_data, self.fil_size, 1)
        if self.DropoutFlag:
            in_data = self.drop(in_data, self.DropRate)

        in_data = self.combine(short_1, in_data, in_channels=self.fil_size*2, out_channels=self.fil_size)
        color_logit = self.color_logit(in_data)

        in_data = self.psp(in_data, in_channels=self.fil_size)
        out = self.out(in_data)

        if self.MultiScale:
            # noinspection PyUnboundLocalVariable
            return output3, output4, output5, out  # output1, output2,
        elif self.MultiTaskPar:
            # Define your multi-tasking parallel logic here
            return
        elif self.MultiTaskSeq:
            # Define your multitasking sequential logic here
            return
        else:
            return out