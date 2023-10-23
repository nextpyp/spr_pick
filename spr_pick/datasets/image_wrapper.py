import torch
import numpy as np
import spr_pick

from torch import Tensor

from torch.utils.data import Dataset
from spr_pick.params import NoiseAlgorithm

from enum import Enum, auto
from typing import Union, Dict, Tuple, List, Optional
from numbers import Number
from spr_pick.utils.data_format import DataFormat, DATA_FORMAT_DIM_INDEX, DataDim

NULL_IMAGE = torch.zeros(0)

class DetectionDataset(Dataset):
	INPUT = 0
	TARGET = 1
	HM = 2
	METADATA = -1
	HM_SMALL = 3

	def __init__(
		self,
		child: Dataset,
		enable_metadata:bool = True,
		pad_uniform: bool = False,
		pad_multiple: int = None,
		square: bool = False,
		data_format: str = DataFormat.CHW,
		):
		self.child = child 
		# self.pi = self.child.pi
		self.train_targets = self.child.train_targets
		self.enable_metadata = enable_metadata
		self.pad_uniform = pad_uniform
		self.pad_multiple = pad_multiple
		self.square = square
		self.data_format = data_format
		self._max_image_size = None
		# self.positive_fraction = self.child.positive_fraction
		if self.pad_uniform:
			_ = self.max_image_size

	def __len__(self) -> int:
		return self.child.__len__()

	def __getitem__(self, index:int):
		data = self.child.__getitem__(index)
		if self.child.train:
			img = data[0]
			target = data[1]
			gt = data[2]
			hm = data[3]
			hm_small = data[-3]
			index = data[-2]
			name = data[-1]
			if self.enable_metadata:
				metadata={}
				metadata[DetectionDataset.Metadata.INDEXES] = index 
				metadata[DetectionDataset.Metadata.TARGET] = target 
				metadata[DetectionDataset.Metadata.NAME] = name
				metadata[DetectionDataset.Metadata.HM] = hm
				metadata[DetectionDataset.Metadata.HM_SMALL] = hm_small
				if gt is not None:
					metadata[DetectionDataset.Metadata.GT] = gt
				else:
					metadata[DetectionDataset.Metadata.GT] = []
			else:
				metadata = None
			# (inp, tar, hm, metadata) = self.prepare_input_hm(img, target, hm, metadata)
			(inp, tar, hm, hm_small, metadata) = self.prepare_input_hm_all(img, target, hm, hm_small, metadata)
			if self.enable_metadata:
				return (inp, tar, hm, hm_small, metadata)
			else:
				return (inp, tar, hm, hm_small)

		else:
			img = data[0]
			# print('img', img.shape)
			target = data[1]
			gt = data[2]
			hm = data[3]
			hm_small = data[-3]
			# print('hm small', hm_small.shape)
			index = data[-2]
			name = data[-1]
			if self.enable_metadata:
				metadata = {}
				metadata[DetectionDataset.Metadata.INDEXES] = index
				metadata[DetectionDataset.Metadata.TARGET] = target 
				metadata[DetectionDataset.Metadata.NAME] = name
				metadata[DetectionDataset.Metadata.HM] = hm
				metadata[DetectionDataset.Metadata.HM_SMALL] = hm_small
				if gt is not None:
					metadata[DetectionDataset.Metadata.GT] = gt
				else:
					metadata[DetectionDataset.Metadata.GT] = []
			else:
				metadata = None 
			
			(inp, tar, hm, hm_small, metadata) = self.prepare_input_hm_all(img, target, hm, hm_small, metadata)
			if self.enable_metadata:
				return (inp, tar, hm, hm_small, metadata)
			else:
				return (inp, tar, hm, hm_small)

	def prepare_input_hm_all(self, image: Tensor, target: Tensor, hm: Tensor, hm_small: Tensor, metadata:Dict={}) -> Tuple[Tensor, Tensor, Dict]:
		image_shape = image.shape 

		img_in, target = image, target 

		ref = NULL_IMAGE

		inp, inp_tar, hm = image, target, hm
		# print('inp pre', inp.shape)
		inp = self.pad_to_output_size(inp)
		# print('inp', inp.shape)
		hm = self.pad_to_output_size(hm)
		# print('hm', hm.shape)
		hm_small = self.pad_to_output_size(hm_small)
		if metadata is not None:
			metadata[DetectionDataset.Metadata.IMAGE] = img_in
			metadata[DetectionDataset.Metadata.TARGET] = target
			metadata[DetectionDataset.Metadata.IMAGE_SHAPE] = torch.tensor(image_shape)
			metadata[DetectionDataset.Metadata.HM] = hm
			metadata[DetectionDataset.Metadata.HM_SMALL] = hm_small
		return (inp, inp_tar, hm, hm_small, metadata)

	def prepare_input_hm(self, image: Tensor, target: Tensor, hm: Tensor, metadata:Dict={}) -> Tuple[Tensor, Tensor, Dict]:
		image_shape = image.shape 

		img_in, target = image, target 

		ref = NULL_IMAGE

		inp, inp_tar, hm = image, target, hm
		inp = self.pad_to_output_size(inp)
		hm = self.pad_to_output_size(hm)

		if metadata is not None:
			metadata[DetectionDataset.Metadata.IMAGE] = img_in
			metadata[DetectionDataset.Metadata.TARGET] = target
			metadata[DetectionDataset.Metadata.IMAGE_SHAPE] = torch.tensor(image_shape)
			metadata[DetectionDataset.Metadata.HM] = hm
		return (inp, inp_tar, hm, metadata)

	def prepare_input(self, image: Tensor, target: Tensor, metadata:Dict={}) -> Tuple[Tensor, Tensor, Dict]:
		image_shape = image.shape 

		img_in, target = image, target 

		ref = NULL_IMAGE

		inp, inp_tar = image, target
		inp = self.pad_to_output_size(inp)
		if metadata is not None:
			metadata[DetectionDataset.Metadata.IMAGE] = img_in
			metadata[DetectionDataset.Metadata.TARGET] = target
			metadata[DetectionDataset.Metadata.IMAGE_SHAPE] = torch.tensor(image_shape)
		return (inp, inp_tar, metadata)
	def prepare_input_aug(self, image: Tensor, aug_img: Tensor, target: Tensor, metadata:Dict={}) -> Tuple[Tensor, Tensor, Tensor, Dict]:
		image_shape = image.shape
		img_in, aug_in, target = image, aug_img, target 
		ref = NULL_IMAGE
		inp, aug_inp, inp_tar = image, aug_img, target 
		inp = self.pad_to_output_size(inp)
		aug_inp = self.pad_to_output_size(aug_inp)
		if metadata is not None:
			metadata[DetectionDataset.Metadata.IMAGE] = img_in
			metadata[DetectionDataset.Metadata.AUG_IMG] = aug_img 
			metadata[DetectionDataset.Metadata.IMAGE_SHAPE] = torch.tensor(image_shape)
		return (inp, aug_inp, inp_tar, metadata)




	@property
	def max_image_size(self) -> List[int]:
		""" Find the maximum image size in the dataset. Will try calling `image_size` method
		first in case a fast method for checking size has been implemented. Will fall back
		to loading images from the dataset as normal and checking their shape. Once this
		method has been called once the maximum size will be cached for subsequent calls.
		"""
		if self._max_image_size is None:
			try:
				image_sizes = [self.child.image_size(i) for i in range(len(self.child))]
			except AttributeError:
				image_sizes = [torch.tensor(data[0].shape) for data in self.child]

			image_sizes = torch.stack(image_sizes)
			max_image_size = torch.max(image_sizes, dim=0).values
			self._max_image_size = max_image_size
		return self._max_image_size

	def get_output_size(self, image: Tensor) -> Tensor:
		"""Calculate output size of an image using the current padding configuration.
		"""
		df = DATA_FORMAT_DIM_INDEX[self.data_format]
		# Use largest image size in dataset if returning uniform sized tensors
		if self.pad_uniform:
			image_size = self.max_image_size
		else:
			image_size = image.shape
		image_size = list(image_size)
		# Pad width and height axis up to a supported multiple
		if self.pad_multiple:
			pad = self.pad_multiple
			for dim in [DataDim.HEIGHT, DataDim.WIDTH]:
				image_size[df[dim]] = (image_size[df[dim]] + pad - 1) // pad * pad

		# Pad to be a square
		if self.square:
			size = max(image_size[df[DataDim.HEIGHT]], image_size[df[DataDim.WIDTH]])
			image_size[df[DataDim.HEIGHT]] = size
			image_size[df[DataDim.WIDTH]] = size

		return torch.tensor(image_size)

	def pad_to_output_size(self, image: Tensor):
		""" Apply reflection padding to the image to meet the current padding
		configuration. Note that padding is handled by Numpy.
		"""

		supported = [DataFormat.CHW, DataFormat.CWH, DataFormat.BCHW, DataFormat.BCWH]
		if self.data_format not in supported:
			raise NotImplementedError("Padding not supported by data format")

		df = DATA_FORMAT_DIM_INDEX[self.data_format]
		output_size = self.get_output_size(image)
		# Already correct, do not pad
		if all(output_size == torch.tensor(image.shape)):
			return image

		left, top = 0, 0
		right = output_size[df[DataDim.WIDTH]] - image.shape[df[DataDim.WIDTH]]
		bottom = output_size[df[DataDim.HEIGHT]] - image.shape[df[DataDim.HEIGHT]]
		# Pad Width/Height ignoring other axis
		pad_matrix = [[0, 0]] * len(self.data_format)
		pad_matrix[df[DataDim.WIDTH]] = [left, right]
		pad_matrix[df[DataDim.HEIGHT]] = [top, bottom]
		# PyTorch methods expect PIL images so fallback to Numpy for padding
		np_padded = np.pad(image, pad_matrix, mode="reflect")
		# np_padded_tar = np.pad(target, pad_matrix, mode="constant")
		# Convert back to Tensor
		return torch.tensor(
			np_padded, device=image.device, requires_grad=image.requires_grad
		)

	@staticmethod
	def _unpad_single(image: Tensor, shape: Tensor) -> Tensor:
		# Create slice list extracting from 0:n for each shape axis
		slices = list(map(lambda x: slice(*x), (zip([0] * len(shape), shape))))
		return image[slices]

	@staticmethod
	def _unpad(image: Tensor, shape: Tensor) -> Union[Tensor, List[Tensor]]:
		if len(image.shape) <= shape.shape[-1]:
			return DetectionDataset._unpad_single(image, shape)
		return [DetectionDataset._unpad_single(i, s) for i, s in zip(image, shape)]
	@staticmethod
	def unpad(image: Tensor, target: Tensor, metadata: Dict, batch_index: int=None):
		inp_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
		if batch_index is not None:
			image = image[batch_index]
			target = target[batch_index]
			inp_shape = inp_shape[batch_index]
		return [DetectionDataset._unpad(image, inp_shape), DetectionDataset._unpad(target, inp_shape)]
	def unpad_img(image: Tensor, metadata: Dict, batch_index: int=None):
		inp_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
		if batch_index is not None:
			image = image[batch_index]
			# target = target[batch_index]
			inp_shape = inp_shape[batch_index]
		return DetectionDataset._unpad(image, inp_shape)


	class Metadata(Enum):
		IMAGE = auto()
		HM = auto()
		HM_SMALL = auto()
		IMAGE_SHAPE = auto()
		AUG_IMG = auto()
		INDEXES = auto()
		TARGET = auto()
		GT = auto()
		NAME = auto()
		PROB = auto()