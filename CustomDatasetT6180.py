from typing import Any, Final, List
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from enum import Enum
from numpy import ndarray
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from randaugment import RandAugment
from torchvision import transforms
from torchvision.utils import save_image

class CustomDataset(Dataset):
    # 초기화 및 선언
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        # dir 경로 data\\train\\images
        FILE_FORMAT: Final = data_dir + "*\\*.jpg"
        self.DIR_DELIMITER: Final = '\\'
        self.PROFILE_INDEX: Final = -2
        self.GENDER_INDEX: Final = 1
        self.AGE_INDEX: Final = 3
        self.MASK_INDEX: Final = -1
        self.PROFILE_DELIMITER: Final = '_'
        self.PREFIX: Final = -5
        self.MASK_LABEL_POINT: Final = 6
        self.GENDER_LABEL_POINT: Final = 3
        
        file_paths: Final = glob(FILE_FORMAT)
        self.file_paths = file_paths
        self.labels, self.genders, self.ages = self._setup()
        
        self.transform = self._torchvision_transforms()
        
    def __len__(self) -> int:
        return len(self.images)
    
    # 아직 안 끝남~ 이미지 불어와야 하고 그건 나중에 하자
    def __getitem__(self, index: int) -> Any:
        image = Image.open(self.file_paths[index]) 
        image = self.transform(image)
        
        age_label: Final = self.ages[index]
        mask_label: Final = self.labels[index]
        gender_label: Final = self.genders[index]
        
        return image, self._get_multi_class_labels(age_label, mask_label, gender_label)
    
    def _setup(self) -> List[int]:
        ages = []
        genders = []
        labels = []
        
        for image in self.file_paths:
            images_each_split_dir = image.split(self.DIR_DELIMITER)
            profile = images_each_split_dir[self.PROFILE_INDEX]
            
            profile_split = profile.split(self.PROFILE_DELIMITER)
            gender = profile_split[self.GENDER_INDEX]
            age = profile_split[self.AGE_INDEX]
            
            # mask1, 2,,,을 같은 라벨로 보기 위해 -5로 지정
            label = images_each_split_dir[self.MASK_INDEX][:self.PREFIX]
            
            ages.append(age)
            labels.append(label)
            genders.append(gender)
        
        self.labels = self._convert_class_to_numeric(labels)
        self.genders = self._convert_class_to_numeric(genders)
        
        ages = list(map(int, ages))
        self.ages = self._convert_age_to_dummy(ages)
        
        return self.labels.copy(), self.genders.copy(), self.ages.copy()
    
    # 유틸 메서드    
    def _convert_class_to_numeric(self, class_array: List[str]) -> ndarray:
        class_to_number: Final = {class_value : idx for idx, class_value in enumerate(np.unique(class_array))}
        return np.vectorize(class_to_number.get)(class_array).copy()

    def _convert_age_to_dummy(self, ages: List[int]) -> ndarray:
        return np.vectorize(Age._label_ages)(ages).copy()
    
    def _get_multi_class_labels(self, mask_label: int, gender_label: int, age_label: int) -> int:
        """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
        return mask_label * self.MASK_LABEL_POINT + gender_label * self.GENDER_LABEL_POINT + age_label
    
    def _album_transforms(self):
        return A.Compose([
            A.RandomResizedCrop(height=150, width=300, scale=(0.3, 1.0)),
            A.Resize(100, 100),
            A.OneOf([
                A.VerticalFlip(p=1)
            ], p = 0.5),
            ToTensorV2()
        ])
        
    def _torchvision_transforms(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(150, scale=(0.3, 1.0)),
            transforms.Resize((100, 100)),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class Age(int, Enum):
    """Age Enum"""
    
    YOUNG = 0
    MIDDLE = 1
    OLD = 2
    
    @classmethod
    def _label_ages(cls, ages):
        if(ages < 30):
            return cls.YOUNG
        elif(ages < 60):
            return cls.MIDDLE
        else:
            return cls.OLD

# if(__name__ == "__main__"):
#     dataset = CustomDataset("data\\train\\images\\")
#     data = dataset.__getitem__(0)
#     print(data[0].shape, data[1])
    
#     save_image(data[0], "result.jpg")
