import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        <table style="border: 2px;">
            <tr>
                <td colspan="3" align="center"> Annotated colors in each label 
            </td>
            </tr><tr>
                <td align="center"> Class </td>
                <td align="center"> Grayscale </td>
            </tr><tr>
                <td align="center"> Others </td>
                <td align="center"> 0 </td>  0
            </tr><tr>
                <td align="center"> Sky </td>
                <td align="center"> 50 </td> 1
            </tr><tr>
                <td align="center"> Land </td>
                <td align="center"> 100 </td> 2
            </tr>
            <tr>
                <td align="center"> Sea Objects </td>
                <td align="center"> 150 </td> 3
                </tr>
        </table>
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext)) # img.shape  (512, 512, 3)

        mask = []
        for i in range(self.num_classes):

            # print(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext))

            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask) # mask.shape (512, 512, 4)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1) # img.shape (3, 512, 512)
        mask = mask.astype('float32') / 255 #np.max(mask) 1.0 and np.min(mask) 0.0
        mask = mask.transpose(2, 0, 1)  # mask.shape (4, 512, 512)

        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}
