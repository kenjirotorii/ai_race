from torch.utils.data import Dataset

from utils import get_img


class AEDataSet(Dataset):
    def __init__(self, home_path, path_list, transform=None, crop=True):
        """DataSet for Autoencoder

        Args:
            home_path (str): home dir path
            path_list (list): image file names
            transform (transforms): transforms
            crop (bool): crop images or not
        """
        self.home_path = home_path
        self.path_list = path_list
        self.transform = transform
        self.crop = crop
        self.seasons = ["spring", "summer", "autumn", "winter"]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        input_paths = ["{}/{}/images/{}".format(self.home_path, s, self.path_list[idx]) for s in self.seasons]
        inputs = [get_img(path, self.crop) for path in input_paths]
        target_path = self.home_path + "/normal/images/" + self.path_list[idx]
        target = get_img(target_path, self.crop)
        
        if self.transform:
            inputs = [self.transform(img.copy()) for img in inputs]
            target = self.transform(target.copy())

        return inputs, target


class ControlDataSet(Dataset):
    def __init__(self, img_df, transform=None, crop=True):
        self.img_df = img_df
        self.crop = crop
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        label = self.img_df.iat[idx, 2]
        img_path = self.img_df.iat[idx, 1]

        img = get_img(img_path, self.crop)

        if self.transform:
            img = self.transform(img.copy())

        return img, label
