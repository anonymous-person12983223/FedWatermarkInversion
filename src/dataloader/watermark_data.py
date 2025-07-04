import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from src.utils.image_utils import trigger_image, normalize


class WatermarkDataset(Dataset):
    """
    A custom PyTorch Dataset used to generate synthetic images with watermark patterns or abstract backgrounds.
    """

    WATERMARK_PATH = "data/watermarks"
    WATERMARK_28x28 = "MPATTERN"
    WATERMARK_32x32 = "CPATTERN"
    ABSTRACT_32x32 = "ABSTRACT_32x32"
    ABSTRACT_28x28 = "ABSTRACT_28x28"

    def __init__(self, args, image_size=(32, 32), num_classes=10, grayscale=False, normalize=True):
        """
        Initialize the dataset with configuration parameters.

        Args:
            args (Namespace): Configuration object with fields like wm_size, seed, background, pattern, etc.
            image_size (tuple): Desired size of the output image (default: (32, 32)).
            num_classes (int): Number of output classes.
            grayscale (bool): Whether to use grayscale images (1 channel) or color (3 channels).
            normalize (bool): Whether to normalize pixel values (usually to zero mean/unit variance).
        """
        self.normalize = normalize
        self.background = args.background
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_images = args.wm_size
        self.grayscale = grayscale
        self.args = args
        self.labels = self.gen_labels()

        # Load abstract backgrounds if applicable
        if self.args.background == "abstract":
            if image_size[0] == 32:
                abstract_dir = os.path.join(self.WATERMARK_PATH, self.ABSTRACT_32x32)
            else:
                abstract_dir = os.path.join(self.WATERMARK_PATH, self.ABSTRACT_28x28)
            abstract_files = sorted([os.path.join(abstract_dir, f)
                                     for f in os.listdir(abstract_dir)
                                     if f.endswith(".jpg") or f.endswith(".png")])
            random_state = np.random.RandomState(self.args.seed)
            random_state.shuffle(abstract_files)
            self.abstract_files = abstract_files

        # Load watermark patterns
        self.pattern = self.args.pattern
        full_wm_path = self.WATERMARK_PATH
        if image_size[0] == 28:
            full_wm_path = os.path.join(full_wm_path, self.WATERMARK_28x28)
        else:
            full_wm_path = os.path.join(full_wm_path, self.WATERMARK_32x32)
        self.pattern_files = sorted([os.path.join(full_wm_path, f) for f in os.listdir(full_wm_path)])

    def __len__(self):
        return self.num_images

    def gen_labels(self):
        """
        Generate random class labels for the dataset.

        Returns:
            np.ndarray: Array of integer labels.
        """
        random_state = np.random.RandomState(self.args.seed)
        numbers = np.arange(0, self.num_classes)
        numbers = np.repeat(numbers, self.num_images // self.num_classes)
        labels = random_state.choice(numbers, size=self.num_images, replace=False)
        return labels

    def get_pattern(self, lbl):
        """
        Load a watermark pattern image corresponding to a given label.

        Args:
            lbl (int): Class label.

        Returns:
            torch.Tensor: Normalized pattern image (H x W x C).
        """
        pattern = np.array(Image.open(self.pattern_files[lbl]))
        pattern = torch.from_numpy(pattern).float() / 255.0
        return pattern

    def get_trigger(self, idx):
        """
        Generate a synthetic image at the given index by combining background and pattern.

        Args:
            idx (int): Index of the image to generate.

        Returns:
            Tuple[torch.Tensor, int]: A tuple of (image tensor, label).
        """
        if idx >= self.num_images:
            raise IndexError("Index out of range")

        channels = 1 if self.grayscale else 3
        seed = idx * self.args.seed
        random_state = np.random.RandomState(seed)

        # Background generation
        if self.background == "random":
            image = random_state.rand(*self.image_size, channels).astype(np.float32)
        elif self.background == "randomN":
            image = random_state.randn(*self.image_size, channels).astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())
        elif self.background == "abstract":
            abstract_file = self.abstract_files[idx]
            with Image.open(abstract_file) as img:
                image = np.array(img, dtype=np.float32) / 255
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
        else:
            image = np.zeros((*self.image_size, channels), dtype=np.float32)

        target_class = self.labels[idx]
        image = torch.from_numpy(image).float()

        # Overlay watermark pattern if enabled
        if self.pattern:
            pattern = self.get_pattern(target_class)
            image = trigger_image(image, pattern, channels=channels)
        else:
            image = image.permute(2, 0, 1)  # Convert to CHW

        if self.normalize:
            normalize(image, channels)

        return image, target_class

    def __getitem__(self, idx):
        return self.get_trigger(idx)

    def get_loader(self):
        """
        Create a PyTorch DataLoader for this dataset.

        Returns:
            DataLoader: Dataloader with fixed batch size, no shuffling.
        """
        batch_size = self.args.batch_size if "batch_size" in self.args else self.args.unlearn_batch_size
        return torch.utils.data.DataLoader(self, batch_size=batch_size,
                                           shuffle=False)

    def get_class_sample(self, label, size, batch_size):
        """
        Return a DataLoader containing `size` number of images of the given class label.

        Args:
            label (int): Target class label.
            size (int): Number of samples to retrieve.
            batch_size (int): Batch size for returned DataLoader.

        Returns:
            DataLoader: Subset DataLoader with only the specified class samples.
        """
        subset_indices = []
        batch_size = batch_size or self.args.batch_size

        for index in range(len(self)):
            img, lbl = self[index]
            if label == lbl:
                subset_indices.append(index)
                if len(subset_indices) == size:
                    break

        subset = torch.utils.data.Subset(self, subset_indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                           shuffle=False, num_workers=1)

