import torch
import spr_pick
import torch.nn as nn

from torch import Tensor

from spr_pick.models.utility import Shift2d
from spr_pick.models.feature_extractor import ResNet, ResNet6, ResNet8
from spr_pick.models.classifier import LinearClassifier
import torch.nn.functional as F
class DualNetworkShallow(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        blindspot: bool = False,
        detect: bool = False,
        detect_out_channels: int = 1,
        zero_output_weights: bool = False,
    ):
        super(DualNetworkShallow, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.Conv2d = ShiftConv2d if self.blindspot else nn.Conv2d
        self.detect = detect

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            if blindspot:
                return nn.Sequential(Shift2d((1, 0)), max_pool)
            return max_pool

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            self.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(48),
            self.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(48),
            _max_pool_block(nn.MaxPool2d(2)),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                self.Conv2d(48, 48, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                # nn.BatchNorm2d(48),
                _max_pool_block(nn.MaxPool2d(2)),
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        # self.encode_block_4 = _encode_block_2_3_4_5()
        # self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            self.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.InstanceNorm2d(48),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(96),
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(96),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                self.Conv2d(144, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                # nn.BatchNorm2d(96),
                self.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                # nn.BatchNorm2d(96),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        # self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            self.Conv2d(96 + in_channels, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(96),
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(96),
        )

        ####################################
        # Output Block
        ####################################

        if self.blindspot:
            # Shift 1 pixel down
            self.shift = Shift2d((1, 0))
            # 4 x Channels due to batch rotations
            nin_a_io = 384
        else:
            nin_a_io = 96

        
        
        # self.output_detect = nn.Sequential(
        #     self.Conv2d(nin_a_io, 96, 1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.BatchNorm2d(96),
        #     self.Conv2d(96, 1, 1),
        #     nn.BatchNorm2d(96)
        #     )
        # self.batchnorm = nn.BatchNorm2d(96)
        self.output_block = nn.Sequential(
            self.Conv2d(nin_a_io, nin_a_io, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(nin_a_io),
            self.Conv2d(nin_a_io, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(96)
        )
        self.detect_block = nn.Sequential(
            self.Conv2d(nin_a_io, nin_a_io, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(nin_a_io),
            self.Conv2d(nin_a_io, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.BatchNorm2d(96)
        )
        # if self.detect:
            # self.detect_block = nn.Sequential(
            #   nn.Conv2d(nin_a_io, nin_a_io, 1),
            #   nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            #   nn.Conv2d(nin_a_io, 96, 1),
            #   nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            #   nn.Conv2d(96, detect_out_channels,1),
            #   )
            # self.bn = nn.BatchNorm2d(2)
            # # self.bn = nn.InstanceNorm2d(2)
            # self.detect_block =LinearClassifier(ResNet8(units=96, bn=True))
            # print('self.detect_block')
            # print(self.detect_block)
        self.output_conv = self.Conv2d(96, out_channels, 1)
        # self.output_conv_s = nn.Conv2d(out_channels, 1, 3, padding=2, dilation=2)
        # self.output_conv_f = nn.Conv2d(96, 1, 7, padding=3)
        self.output_conv_f = nn.Conv2d(96, 1, 1)
        # self.output_conv_s = nn.Conv2d(96, 1, 7, stride=2, padding=3)
        # self.detect_block = LinearClassifier(ResNet8(units=32, bn=False))

        # Initialize weights
        self.init_weights()

    @property
    def blindspot(self) -> bool:
        return self._blindspot

        
    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
        # Initialise last output layer
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def fill(self, stride = 1):
        return self.detect_block.fill(stride=stride)

    def unfill(self):
        return self.detect_block.unfill()


    def forward(self, x: Tensor):
        orig_inp = x
        # print('orig', x.shape)
        if self.blindspot:
            rotated = [spr_pick.utils.rotate(x, rot) for rot in (0, 90, 180, 270)]
            x = torch.cat((rotated), dim=0)
        # print('cat x:', x.shape)
        pool1 = self.encode_block_1(x)
        # print('poo1', pool1.shape)
        pool2 = self.encode_block_2(pool1)
        # print('poo2', pool2.shape)
        pool3 = self.encode_block_3(pool2)
        # print('poo3', pool3.shape)
        # pool4 = self.encode_block_4(pool3)
        # pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool3)
        # print('enc', encoded.shape)

        # Decoder
        upsample5 = self.decode_block_6(encoded)
        # print('u5', upsample5.shape)
        # concat5 = torch.cat((upsample5, pool4), dim=1)
        concat3 = torch.cat((upsample5, pool2), dim=1)
        # print('concat3')
        # print(concat3.shape)
        # upsample4 = self.decode_block_5(concat5)
        # print('pool3')
        # print(pool3.shape)
        # concat4 = torch.cat((upsample4, pool3), dim=1)
        # print('concat4')
        # print(concat4.shape)
        # upsample3 = self.decode_block_4(concat4)
        # concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_5(concat3)
        # print('u2', upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        # print('c2', concat2.shape)
        # print
        # detect_out_1 = self.detect_block(concat2)
        # if self.blindspot:
        #     # shifted = self.shift(detect_out_1)
        #     rotated_batch = torch.chunk(detect_out_1, 4, dim=0)
        #     aligned = [
        #         joint.utils.rotate(rotated, rot)
        #         for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
        #     ]
        #     detect_out_1 = torch.cat(aligned, dim=1)
        #     if self.detect:
        #         detect_out = self.output_detect(detect_out_1)
                # print('detect_out', detect_out.shape)
        upsample1 = self.decode_block_2(concat2)
        # print('up1,', upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)
        # print('decode x', x.shape)
        if self.blindspot:
            # Apply shift
            shifted = self.shift(x)
            # Unstack, rotate and combine
            rotated_batch = torch.chunk(shifted, 4, dim=0)
            aligned = [
                spr_pick.utils.rotate(rotated, rot)
                for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
            ]
            x_blind = torch.cat(aligned, dim=1)

            x_blind = self.output_block(x_blind)
            if self.detect:
                x_out = self.output_conv(x_blind)
                rotated_batch = torch.chunk(x, 4, dim=0)
                aligned = [
                    spr_pick.utils.rotate(rotated, rot)
                    for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
                ]
                x_detect = torch.cat(aligned, dim=1)
                x_detect = self.detect_block(x_detect)
                detect_out = self.output_conv_f(x_detect)

                # fm = self.output_conv_f(x)
                # x_cat = torch.cat((x, x_out), dim=1)
                # hm = self.output_conv_f(x_cat)
                # hm_s = self.output_conv_s(x)
                # print('hm_s', hm_s.shape)
                # print('x', x.shape)
                # return x_out, hm
                return x_out, detect_out
        else:
            x = self.output_block(x)
            x = self.output_conv(x)
            # print('x', x.shape)
            return x 
            
    @staticmethod
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.

        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 3
        return 2 ** max_pool_layers
# from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InPlaceUp(nn.Module):
    def __init__(self, in_channels):
        super(InPlaceUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, in_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class segmentation_module(nn.Module):
    def __init__(self, curr_dim, **kwargs):
        super(segmentation_module, self).__init__()

        self.n_channels = curr_dim
        self.n_classes = 1

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.up5 = InPlaceUp(64)
        self.up6 = InPlaceUp(64)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        logits = self.outc(x)
        return logits

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        # pad = 0
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('x',x.shape)
        conv = self.conv(x)
        # print('convx,', conv.shape)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu
# nn.Upsample(scale_factor=2, mode="nearest"),
def make_saliency_layer(curr_dim):
    return nn.Sequential(convolution(3, curr_dim, curr_dim*8, with_bn=False),
        # nn.Upsample(scale_factor=2, mode="nearest"),
        convolution(3, curr_dim*8, curr_dim*16, with_bn=False),
        nn.Upsample(scale_factor=2, mode="bilinear"),
        # nn.ConvTranspose2d(curr_dim*16, curr_dim*16, 3, 2),
        # nn.ReLU(inplace=True),
        convolution(3, curr_dim*16, curr_dim*32, with_bn = False),
        nn.Conv2d(curr_dim*32, 1, (1,1)),
        # nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear"),
        # nn.ConvTranspose2d(1, 1, 3, 2),
        # nn.ReLU(inplace=True), 
        )
class JointNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        blindspot: bool = False,
        detect: bool = False,
        detect_out_channels: int = 1,
        zero_output_weights: bool = False,
    ):
        super(JointNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.denoise_branch = DenoiseNetwork(in_channels = in_channels, out_channels = out_channels, blindspot = blindspot, detect= detect, detect_out_channels=detect_out_channels, zero_output_weights=zero_output_weights)
        self.mask = make_saliency_layer(in_channels)
        # self.mask = segmentation_module(in_channels)
    def reparameterize(self, x):
        mu_x = x[:,0:self.in_channels,...]
        A_c = x[:, self.in_channels:self.out_channels,...]
        sigma_x = A_c ** 2
        epsilon = torch.randn_like(mu_x)
        z = mu_x + epsilon * sigma_x 
        return z 
    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
        # Initialise last output layer
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.

        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers
        
    def forward(self,x):
        # print('x,', x.shape)
        out_stats = self.denoise_branch(x)
        # print('out_stats', out_stats.shape)
        sampled_img = self.reparameterize(out_stats)
        # print('sampled stats,', torch.min(sampled_img), torch.max(sampled_img), torch.mean(sampled_img))
        probs = self.mask(sampled_img)
        probs = nn.MaxPool2d(4)(probs)
        # mins_prob = torch.min(probs.reshape(probs.shape[0],probs.shape[1],probs.shape[2]*probs.shape[3]), dim=2)
        # maxs_prob = torch.max()
        # m = nn.Sigmoid()
        # probs = m(probs)
        # prob_pos = nn.Threshold(0.5,0)(probs)
        # prob_bg = nn.Threshold(0.5,0)(1-probs)
        # fg = sampled_img * probs
        # # bg = sampled_img * (1-probs)
        # bg = sampled_img - fg
        return out_stats,probs,sampled_img










class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.detector = LinearClassifier(ResNet8(bn=True))
        self.m = nn.BatchNorm2d(1)
    def fill(self, stride = 1):
        return self.detector.fill(stride=stride)

    def unfill(self):
        return self.detector.unfill()

    def forward(self, x):
        # m = nn.Norm(x.size()[1:], elementwise_affine=False)
        # m = nn.BatchNorm2d(1)
        x = self.m(x)
        out = self.detector(x)
        return out



class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x


