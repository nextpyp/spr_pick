import torch
import ssdn
import torch.nn as nn

from torch import Tensor

from ssdn.models.utility import Shift2d
from ssdn.models.feature_extractor import ResNet, ResNet6, ResNet8
from ssdn.models.classifier import LinearClassifier
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
			# nn.InstanceNorm2d(48),
			self.Conv2d(48, 48, 3, padding=1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			# nn.InstanceNorm2d(48),
			_max_pool_block(nn.MaxPool2d(2)),
		)

		# Layers: enc_conv(i), pool(i); i=2..5
		def _encode_block_2_3_4_5() -> nn.Module:
			return nn.Sequential(
				self.Conv2d(48, 48, 3, stride=1, padding=1),
				nn.LeakyReLU(negative_slope=0.1, inplace=True),
				# nn.InstanceNorm2d(48),
				_max_pool_block(nn.MaxPool2d(2)),
			)

		# Separate instances of same encode module definition created
		self.encode_block_2 = _encode_block_2_3_4_5()
		self.encode_block_3 = _encode_block_2_3_4_5()
		self.encode_block_4 = _encode_block_2_3_4_5()
		self.encode_block_5 = _encode_block_2_3_4_5()

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
			# nn.InstanceNorm2d(96),
			self.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			# nn.InstanceNorm2d(96),
			nn.Upsample(scale_factor=2, mode="nearest"),
		)

		# Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
		def _decode_block_4_3_2() -> nn.Module:
			return nn.Sequential(
				self.Conv2d(144, 96, 3, stride=1, padding=1),
				nn.LeakyReLU(negative_slope=0.1, inplace=True),
				# nn.InstanceNorm2d(96),
				self.Conv2d(96, 96, 3, stride=1, padding=1),
				nn.LeakyReLU(negative_slope=0.1, inplace=True),
				# nn.InstanceNorm2d(96),
				nn.Upsample(scale_factor=2, mode="nearest"),
			)

		# Separate instances of same decode module definition created
		self.decode_block_4 = _decode_block_4_3_2()
		self.decode_block_3 = _decode_block_4_3_2()
		self.decode_block_2 = _decode_block_4_3_2()

		# Layers: dec_conv1a, dec_conv1b, dec_conv1c,
		self.decode_block_1 = nn.Sequential(
			self.Conv2d(96 + in_channels, 96, 3, stride=1, padding=1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			# nn.InstanceNorm2d(96),
			self.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			# nn.InstanceNorm2d(96),
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

		
		
		self.output_block = nn.Sequential(
			self.Conv2d(nin_a_io, nin_a_io, 1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			# nn.InstanceNorm2d(nin_a_io),
			self.Conv2d(nin_a_io, 96, 1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			# nn.InstanceNorm2d(96)
		)
		if self.detect:
			# self.detect_block = nn.Sequential(
			# 	nn.Conv2d(nin_a_io, nin_a_io, 1),
			# 	nn.LeakyReLU(negative_slope = 0.1, inplace=True),
			# 	nn.Conv2d(nin_a_io, 96, 1),
			# 	nn.LeakyReLU(negative_slope = 0.1, inplace=True),
			# 	nn.Conv2d(96, detect_out_channels,1),
			# 	)
			self.bn = nn.BatchNorm2d(3)
			self.detect_block =LinearClassifier(ResNet8(units=96, bn=True))
			# print('self.detect_block')
			# print(self.detect_block)
		self.output_conv = self.Conv2d(96, out_channels, 1)


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
		if self.blindspot:
			rotated = [ssdn.utils.rotate(x, rot) for rot in (0, 90, 180, 270)]
			x = torch.cat((rotated), dim=0)

		pool1 = self.encode_block_1(x)
		pool2 = self.encode_block_2(pool1)
		pool3 = self.encode_block_3(pool2)
		pool4 = self.encode_block_4(pool3)
		pool5 = self.encode_block_5(pool4)
		encoded = self.encode_block_6(pool5)

		# Decoder
		# print('encoded shape')
		# print(encoded.shape)
		upsample5 = self.decode_block_6(encoded)
		concat5 = torch.cat((upsample5, pool4), dim=1)
		# print('concat5')
		# print(concat5.shape)
		upsample4 = self.decode_block_5(concat5)
		# print('pool3')
		# print(pool3.shape)
		concat4 = torch.cat((upsample4, pool3), dim=1)
		# print('concat4')
		# print(concat4.shape)
		upsample3 = self.decode_block_4(concat4)
		concat3 = torch.cat((upsample3, pool2), dim=1)
		upsample2 = self.decode_block_3(concat3)
		concat2 = torch.cat((upsample2, pool1), dim=1)
		upsample1 = self.decode_block_2(concat2)
		concat1 = torch.cat((upsample1, x), dim=1)
		x = self.decode_block_1(concat1)

		if self.blindspot:
			# Apply shift
			shifted = self.shift(x)
			# Unstack, rotate and combine
			rotated_batch = torch.chunk(shifted, 4, dim=0)
			aligned = [
				ssdn.utils.rotate(rotated, rot)
				for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
			]
			x = torch.cat(aligned, dim=1)

			x = self.output_block(x)
			if self.detect:
				# y = self.bn(x)
				# det = self.detect_block(y)
				out = self.output_conv(x)
				out_concat = torch.cat((out, orig_inp), dim =1)
				# print('x shape', x.shape)
				# print('out,',out.shape)
				# y_in = x + out[:,0,:,:]
				y = self.bn(out_concat)
				det = self.detect_block(y)
				return det, out
			else:
				x = self.output_conv(x)
				return x 
		else:
			x = self.output_block(x)
			x = self.output_conv(x)
			# print('x', x.shape)
			return x 

			# return x
		# else:
			# if self.detect and dn is not None:
			# 	cat = x + dn 
			# 	det = self.detect_block(cat)
			# 	x = self.output_block(x)
			# 	return det, x 
			# if dn == None:
			# 	x = self.output_block(x)
			# 	return x 
	@staticmethod
	def input_wh_mul() -> int:
		"""Multiple that both the width and height dimensions of an input must be to be
		processed by the network. This is devised from the number of pooling layers that
		reduce the input size.

		Returns:
			int: Dimension multiplier
		"""
		max_pool_layers = 5
		return 2 ** max_pool_layers

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


