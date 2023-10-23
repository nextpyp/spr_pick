import math
import torch 
from PIL import Image 
from torch import Tensor 
import numpy as np
from typing import Tuple, List, Optional, Any
import torchvision.transforms.functional as F 
import numbers
import random
import warnings
from spr_pick.utils.image import draw_umich_gaussian

try:
    import accimage
except ImportError:
    accimage = None

def _assert_image_tensor(img):
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
def _get_image_size(img: Any) -> List[int]:
    if _is_pil_image(img):
        return img.size
    else:
    	return [img.shape[-1], img.shape[-2]]
    #raise TypeError("Unexpected type {}".format(type(img)))

def crop_mask(target, crop_size):
	# print('target')
	# print(target.shape)

	b, r, c = target.shape
	mask = np.zeros((r, c))
	labeled_ind = np.where(target == 1)
	# print(labeled_ind)
	row_ind, col_ind = labeled_ind[1], labeled_ind[2]
	num_of_labeled = row_ind.shape[0]
	# print(num_of_labeled)
	for i in range(num_of_labeled):
		# mask[max(0, row_ind[i] - crop_size//2):min(r, row_ind[i]+crop_size//2), max(0, col_ind[i]-crop_size//2):min(c, col_ind[i]+crop_size//2)] = 1
		if row_ind[i] - crop_size//2 >= 0 and col_ind[i] - crop_size//2 >=0:
			mask[row_ind[i] - crop_size//2, col_ind[i] - crop_size//2] = 1
	return mask

def _setup_size(size, error_msg):
	if isinstance(size, numbers.Number):
		return int(size), int(size)

	if isinstance(size, Sequence) and len(size) == 1:
		return size[0], size[0]

	if len(size) != 2:
		raise ValueError(error_msg)

	return size

class MyRandomCrop(torch.nn.Module):
	@staticmethod
	def get_params(img: Tensor, output_size: Tuple[int, int], possible_area: Tuple, num_of_possibilities: int, selective: bool) -> Tuple[int, int, int, int]:
		"""Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image or Tensor): Image to be cropped.
			output_size (tuple): Expected output size of the crop.
			possible_area: area that can be cropped
			num_of_possibilities: number of coordinates that can be cropped

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		w, h = _get_image_size(img)
		th, tw = output_size

		if h + 1 < th or w + 1 < tw:
			raise ValueError(
				"Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
			)

		if w == tw and h == th:
			return 0, 0, h, w
		if selective:
			#print('selective crop')
			
			cannot_use = True
			while cannot_use:
				crop_ind = torch.randint(0, num_of_possibilities, size=(1,)).item()
				i = possible_area[0][crop_ind]
				j = possible_area[1][crop_ind]
				if i < (h-th+1) and j < (w-tw+1):
					cannot_use = False
		else:

			i = torch.randint(0, h - th + 1, size=(1, )).item()
			j = torch.randint(0, w - tw + 1, size=(1, )).item()
		return i, j, th, tw


	def __init__(self, size, padding=None, pad_if_needed=False, fill = 0, labeled_only=False, Unlabeled_only = False, ss_mode = False, padding_mode="constant"):
		super().__init__()
		self.size = tuple(_setup_size(
			size, error_msg="Please provide only two dimensions (h, w) for size."
		))
		#print("using custom crop")
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.fill = fill
		self.padding_mode = padding_mode
		# self.labels = labels
		self.ss_mode = ss_mode
		self.labeled_only = labeled_only
		self.Unlabeled_only = Unlabeled_only

	def forward(self, img, labels, hms):
		#labels as numpy array
		#print(img.shape)
		#print(labels)
		#print(labels.shape)
		#print(torch.max(labels))

		labeled_areas = crop_mask(labels, self.size[0])
		non_labeled = np.where(labeled_areas == 0)
		labeled = np.where(labeled_areas == 1)
		num_of_labeled = labeled[0].shape[0]
		# print('num_of_labeled')
		# print(num_of_labeled)
		num_of_nonlabeled = non_labeled[0].shape[0]
		#lb = F.to_tensor(labels)
		lb = labels
		if self.padding is not None:
			img = F.pad(img, self.padding, self.fill, self.padding_mode)
			lb = F.pad(lb, self.padding, self.fill, "constant")
			hms = F.pad(hms, self.padding, self.fill, "constant")
		width, height = _get_image_size(img)
		# pad the width if needed
		if self.pad_if_needed and width < self.size[1]:
			padding = [self.size[1] - width, 0]
			img = F.pad(img, padding, self.fill, self.padding_mode)
			lb = F.pad(lb, padding, self.fill, "constant")
			hms = F.pad(hms, self.padding, self.fill, "constant")
		# pad the height if needed
		if self.pad_if_needed and height < self.size[0]:
			padding = [0, self.size[0] - height]
			img = F.pad(img, padding, self.fill, self.padding_mode)
			lb = F.pad(lb, padding, self.fill, "constant")
			hms = F.pad(hms, self.padding, self.fill, "constant")
		if self.labeled_only:
			i, j, h, w = self.get_params(img, self.size, labeled, num_of_labeled, selective=True)
		elif self.Unlabeled_only:
			i, j, h, w = self.get_params(img, self.size, non_labeled, num_of_nonlabeled, selective=True)
		elif self.ss_mode:
			p = np.random.rand()
			if p <= 0.5:
				i, j, h, w = self.get_params(img, self.size, labeled, num_of_labeled, selective=True)
			else:
				i, j, h, w = self.get_params(img, self.size, labeled, num_of_labeled, selective=False)
		else:
			i, j, h, w = self.get_params(img, self.size, labeled, num_of_labeled, selective=False)
		lb_cropped = F.crop(lb, i, j, h, w)
		hm_cropped = F.crop(hms, i, j, h, w)
		# print('lb_cropped shape', lb_cropped.shape)
		# print(lb_cropped)
		lb = lb_cropped[0, self.size[0]//2, self.size[1]//2]
		new_label = np.zeros(lb_cropped.shape)

		if lb > 0:
			# print('big')
			draw_umich_gaussian(new_label[0], (self.size[0]//2, self.size[1]//2), self.size[0]//4)

		new_label = torch.Tensor(new_label)
		# print('new_label,',new_label.shape)
		# print('new_label', new_label[0,self.size[0]//2-10:self.size[0]//2+10, self.size[0]//2-10:self.size[0]//2+10])

		# print('label,', lb)
		return F.crop(img, i, j, h, w), hm_cropped, hm_cropped




