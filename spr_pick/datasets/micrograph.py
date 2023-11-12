import torch
import torchvision.transforms.functional as F
import os
import glob
import tempfile
import string
import imagesize
import spr_pick
import sys
from torch import Tensor
from spr_pick.utils.transforms import Transform
from spr_pick.utils.data_format import DataFormat, PIL_FORMAT, permute_tuple
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from PIL import Image
from typing import List, Tuple
from spr_pick.utils.loader import load_images_from_list
import pandas as pd 
from spr_pick.utils import files as file_utils
from spr_pick.utils.coordinates import match_coordinates_to_images
import numpy as np
import torchvision.transforms as transforms
import os 


class MicrographDataset(Dataset):
	INPUT = 0 
	TARGET = 1
	METADATA = 2
	AUG_INP = 3
	def __init__(
		self,
		image_path: str,
		coordinate_path: str,
		radius: int, 
		train: bool = True,
		crop: int = 72,
		transform: Transform = None,
		gt_path: str = None,
		augment: Transform = None,
		output_format: str = DataFormat.CHW,
		channels: int= 1,
		bb: int = 32, 
		):
		super(MicrographDataset, self).__init__()
		self.image_path = image_path
		self.coordinate_path = coordinate_path
		self.loader = default_loader
		self.transform = transform
		self.augment = augment
		self.output_format = output_format
		self.gt_path = gt_path
		self.channels = channels
		self.train = train
		self.crop = crop 
		self.train_images, self.train_targets, self.gts, self.hms, self.hms_small, self.num_of_images, self.image_names, self.num_positive_regions, self.total_regions = self.load_data(radius, bb=bb)



	def __getitem__(self, index: int):
		if self.train:
			h = index 
			g = h//2**56
			h = h - g*2**56
			i = h//2**32
			h = h - i*2**32 
			coord = h 

			img = self.train_images[g][i]
			hm = self.hms[g][i]
			# L = torch.from_numpy(self.train_targets[g][i].ravel()).unsqueeze(1)
			# label = L[coord].float()
			# print('label', label)
			target = self.train_targets[g][i].astype(np.float32)
			name = self.image_names[g][i]
			hm = self.hms[g][i]
			label = torch.from_numpy(hm.ravel()).unsqueeze(1)[coord]
			# print('hm unravel', torch.from_numpy(hm.ravel()).unsqueeze(1)[coord])
			hm_small = self.hms_small[g][i]
			if len(self.gts) > 0:
				gt = self.gts[g][i]
			else:
				gt = None

		
			if not isinstance(target, torch.Tensor):
				target = torch.tensor(np.expand_dims(target, axis=0))
				hm = torch.tensor(np.expand_dims(hm, axis=0))
				hm_small = torch.tensor(np.expand_dims(hm_small, axis=0))
			x = coord % img.width 
			y = coord // img.width
			xmi = x - self.crop//2
			xmi_s = xmi//2 
			xma = xmi + self.crop  
			xma_s = xmi_s + self.crop//2
			ymi = y - self.crop//2
			ymi_s = ymi//2
			yma = ymi + self.crop 
			yma_s = ymi_s + self.crop//2
			img = img.crop((xmi, ymi, xma, yma))
			hm = F.crop(hm, xmi,ymi, self.crop, self.crop)
			hm_small = F.crop(hm_small, xmi_s, ymi_s, self.crop//2, self.crop//2)
			# label = F.crop(target, xmi, ymi, self.crop, self.crop)

			

			if self.augment is not None:
				img = self.augment(img)
				hm = self.augment(hm)
				hm_small = self.augment(hm_small)
				# label = self.augment(label)
			if not isinstance(img, torch.Tensor):
				img = F.to_tensor(img)
			if gt is not None:
				if not isinstance(gt, torch.Tensor):
					gt = F.to_tensor(gt)

			if self.output_format is not None:
				img = img.permute(permute_tuple(PIL_FORMAT, self.output_format))
				# if len(label.shape) > 0:
				# 	label = label.permute(permute_tuple(PIL_FORMAT, self.output_format))
			return img, label, gt, hm, hm_small, index, name





		if not self.train:
			img = self.train_images[0][index]
			hm = self.hms[0][index]
			hm_small = self.hms_small[0][index]
			target = self.train_targets[0][index].astype(np.float32)
			name = self.image_names[0][index]
			if len(self.gts) > 0:
				gt = self.gts[0][index]
			else:
				gt = None
			if not isinstance(target, torch.Tensor):
				target = torch.tensor(np.expand_dims(target,axis=0))
				hm = torch.tensor(np.expand_dims(hm, axis=0))
				hm_small = torch.tensor(np.expand_dims(hm_small, axis=0))
			if self.transform:
				img, label, hm= self.transform(img, target, hm)
			else:
				label = target
				hm = hm
				hm_small = hm_small

			if not isinstance(img, torch.Tensor):
				img = F.to_tensor(img)
			if gt is not None:
				if not isinstance(gt, torch.Tensor):
					gt = F.to_tensor(gt)

			if self.output_format is not None:
				img = img.permute(permute_tuple(PIL_FORMAT, self.output_format))
				if gt is not None:
					gt = gt.permute(permute_tuple(PIL_FORMAT, self.output_format))
				if len(label.shape) > 0:

					label = label.permute(permute_tuple(PIL_FORMAT, self.output_format))
					hm = hm.permute(permute_tuple(PIL_FORMAT, self.output_format))
					hm_small = hm_small.permute(permute_tuple(PIL_FORMAT, self.output_format))
			return img, label, gt, hm, hm_small, index, name


	def match_images_targets(self, images, targets, radius, bb, gt_images = None):
		if gt_images is not None:
			matched = match_coordinates_to_images(targets, images, gt_images = gt_images, radius=radius, bb=bb)
		else:
			matched = match_coordinates_to_images(targets, images, radius=radius, bb=bb)
		## unzip into matched lists
		images = []
		targets = []
		hms = []
		gts = []
		names = []
		hms_small = []
		for key in matched:
			these_names = list(matched[key].keys())

			if gt_images is not None:
				these_images, these_gt, these_targets, these_hms, these_hms_small = zip(*list(matched[key].values()))
				gts.append(list(these_gt))
			else:
				these_images,these_targets, these_hms, these_hms_small = zip(*list(matched[key].values()))
			images.append(list(these_images))
			targets.append(list(these_targets))
			hms.append(list(these_hms))
			names.append(these_names)
			hms_small.append(these_hms_small)
		return images,targets, names, gts, hms, hms_small

	def __len__(self):
		return self.num_of_images

	def report_data_stats(self,train_images, train_targets):
		print('source\tsplit\tp_observed\tnum_positive_regions\ttotal_regions')
		num_positive_regions = 0
		total_regions = 0
		for i in range(len(train_images)):
			p = sum(train_targets[i][j].sum() for j in range(len(train_targets[i])))
			p = int(p)
			total = sum(train_targets[i][j].size for j in range(len(train_targets[i])))
			num_positive_regions += p
			total_regions += total
			p_observed = p/total
			p_observed = '{:.3g}'.format(p_observed)
			print(str(i)+'\t'+'train'+'\t'+p_observed+'\t'+str(p)+'\t'+str(total))
		return num_positive_regions, total_regions


	def load_data(self, radius, bb, format_='auto', image_ext=''):
		"""
		Output: train images with [[num_of_images]], (1,num_of_images, dimr, dimc)
		"""
		if os.path.isdir(self.image_path):
			paths = glob.glob(self.image_path+os.sep+'*'+image_ext)
			valid_paths = []
			image_names = []
			for path in paths:
				name = os.path.basename(path)
				name, ext = os.path.splitext(name)
				if ext in ['.mrc','.tiff','.png']:
					image_names.append(name)
					valid_paths.append(path)
			train_images = pd.DataFrame({'image_name': image_names, 'path': valid_paths})
		else:
			train_images = pd.read_csv(self.image_path, sep='\t')
		if self.gt_path is not None:
			if os.path.isdir(self.gt_path):
				paths = glob.glob(self.gt_path+os.sep+'*'+image_ext)
				valid_paths = []
				image_names = []
				for path in paths:
					name = os.path.basename(path)
					name, ext = os.path.splitext(name)
					if ext in ['.mrc','.tiff','.png']:
						image_names.append(name)
						valid_paths.append(path)
				gt_images = pd.DataFrame({'image_name': image_names, 'path': valid_paths})
			else:
				gt_images = pd.read_csv(self.gt_path, sep='\t')
		else:
			gt_images = None

		train_targets = file_utils.read_coordinates(self.coordinate_path, format= format_)
		if 'source' not in train_images and 'source' not in train_targets:
			train_images['source'] = 0
			train_targets['source'] = 0
			if gt_images is not None:
				gt_images['source'] = 0

		train_images = load_images_from_list(train_images.image_name, train_images.path
										, sources=train_images.source)
		if self.gt_path is not None:
			gt_images = load_images_from_list(gt_images.image_name, gt_images.path, sources = gt_images.source)
		else:
			gt_images = None
		names = set()
		for k, d in train_images.items():
			for name in d.keys():
				names.add(name)
		check = train_targets.image_name.apply(lambda x: x in names)

		missing = train_targets.image_name.loc[~check].unique().tolist()

		if False and len(missing) > 0:
			print('{} micrograph(s) listed in the coordinates file are missing from the training images. Image names are listed below.'.format(len(missing)), file=sys.stderr)
			print('Missing micrograph(s) are: {}'.format(missing), file=sys.stderr)

		train_targets = train_targets.loc[check]

		width = 0
		height = 0
		for k, d in train_images.items():
			for image in d.values():
				w, h = image.size
				if w > width:
					width = w 
				if h > height:
					height = h
		out_of_bounds = (train_targets.x_coord > width) | (train_targets.y_coord > height)
		count = out_of_bounds.sum()
		if count > int(0.1*len(train_targets)):
			print('WARNING: {} particle coordinates are out of the micrograph dimensions. Did you scale the micrographs and particle coordinates correctly?'.format(count), file=sys.stderr)

		x_max = train_targets.x_coord.max()
		y_max = train_targets.y_coord.max()
		if x_max < 0.7*width and y_max < 0.7*height:
			print('WARNING: no coordinates are observed with x_coord > {} or y_coord > {}. Did you scale the micrographs and particle coordinates correctly?'.format(x_max, y_max), file=sys.stderr)

		num_micrographs = sum(len(train_images[k]) for k in train_images.keys())
		num_particles = len(train_targets)

		train_images_o, train_targets, train_names, gts_o, hm_o, hm_small_o = self.match_images_targets(train_images, train_targets, radius, bb=bb, gt_images=gt_images)
		num_positive_regions, total_regions = self.report_data_stats(train_images_o, train_targets)

		return train_images_o, train_targets, gts_o, hm_o, hm_small_o, num_micrographs, train_names, num_positive_regions, total_regions

