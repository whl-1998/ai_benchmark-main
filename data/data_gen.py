from torch.utils.data import Dataset as DS
import os
from PIL import Image
from args import args

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(DS):
	def __init__(self, description_path, dataset_path, transform=None):
		with open(description_path, 'r') as f:
			self.env = f.read().split('\n')
		self.transform = transform
		self.dataset_path = dataset_path

	def __len__(self):
		return len(self.env)

	def __getitem__(self, index):
		assert index < len(self), 'index range error'
		img_path, label = self.env[index].strip().split(',')
		img_path = os.path.join(self.dataset_path, img_path)
		img = Image.open(img_path)
		if self.transform is not None:
			img = self.transform(img)
		return img, int(label)

