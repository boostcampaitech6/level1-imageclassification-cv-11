import os
import random
from numpy import ndarray
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Any, Final
from randaugment import RandAugment
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
    Grayscale,
    GaussianBlur,
    RandomHorizontalFlip,
    RandomRotation,
    RandomCrop,
)

# 지원되는 이미지 확장자 리스트
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """파일 이름이 이미지 확장자를 가지는지 확인하는 함수

    Args:
        filename (str): 확인하고자 하는 파일 이름

    Returns:
        bool: 파일 이름이 이미지 확장자를 가지면 True, 그렇지 않으면 False.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    """
    기본적인 Augmentation을 담당하는 클래스

    Attributes:
        transform (Compose): 이미지를 변환을 위한 torchvision.transforms.Compose 객체
    """

    def __init__(self, resize, mean, std, **args):
        """
        Args:
            resize (tuple): 이미지의 리사이즈 대상 크지
            mean (tuple): Normalize 변환을 위한 평균 값
            std (tuple): Normalize 변환을 위한 표준 값
        """
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용

        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지

        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image)


class AddGaussianNoise(object):
    """이미지에 Gaussian Noise를 추가하는 클래스"""

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CustomAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0)),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
    
class AddGaussianNoise_aug:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
                AddGaussianNoise(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)

class CustomAugmentationT6180:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                RandAugment(),
                ToTensor(),
                Normalize(mean=mean, std=std),
                AddGaussianNoise(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)

class CenterCrop_aug:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class GaussianBlur_aug:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class RandomHorizontalFlip_aug:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class ColorJitter_aug:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class RandomRotation_aug:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                RandomRotation(10),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class RandomCrop_aug:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                RandomCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""

    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """성별 라벨을 나타내는 Enum 클래스"""

    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """문자열로부터 해당하는 성별 라벨을 찾아 반환하는 클래스 메서드"""
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    """나이 라벨을 나타내는 Enum 클래스"""

    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        """숫자로부터 해당하는 나이 라벨을 찾아 반환하는 클래스 메서드"""
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    """마스크 데이터셋의 기본 클래스"""

    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()  # 통계시 계산 (평균 및 표준 편차)

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        return self.age_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set




class AugmentationType(int, Enum):
    NONE = -1
    DOUBLE = 0
    OLDAGE = 1
    DOUBLE_OLDAGE = 2

class AugEqualDataset(MaskBaseDataset):
    agument_check = []
    class_labels = []
    class_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[],
        9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[]}
    
    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        super().__init__(data_dir, mean, std, val_ratio)
        resize = (299, 299)
        self.aug_transform = [
                Compose( # double
                    [
                        CenterCrop((320, 256)),
                        Resize(resize, Image.BILINEAR),
                        ToTensor(),
                        Normalize(mean=mean, std=std),
                    ]
                ),
                Compose( # oldage
                    [
                        Resize((299, 299), Image.BILINEAR),
                        ColorJitter(0.5, 0.5, 0.5, 0.5),
                        ToTensor(),
                        Normalize(mean=mean, std=std),
                    ]
                ),
                Compose( # double_oldage
                    [
                        CenterCrop((320, 256)),
                        Resize((299, 299), Image.BILINEAR),
                        ColorJitter(0.5, 0.5, 0.5, 0.5),
                        ToTensor(),
                        Normalize(mean=mean, std=std),
                    ]
                )
            ]
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                class_label = self.encode_multi_class(mask_label, gender_label, age_label)

                for aug_type in range(-1, 1):
                    self.agument_check.append(AugmentationType(aug_type))
                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.class_labels.append(class_label)
                    
                if age_label == AgeLabels.OLD:
                    for _ in range(5):
                        for aug_type_age in range(1, 3):
                            self.agument_check.append(AugmentationType(aug_type_age))
                            self.image_paths.append(img_path)
                            self.mask_labels.append(mask_label)
                            self.gender_labels.append(gender_label)
                            self.age_labels.append(age_label)
                            self.class_labels.append(class_label)

        for i, c in enumerate(self.class_labels):
            self.class_dict[c].append(i)

    def get_old_age_agument(self, index) -> AugmentationType:
        return self.agument_check[index]

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        old_age_agument = self.get_old_age_agument(index)
        
        if old_age_agument != AugmentationType.NONE:
            image_transform = self.aug_transform[old_age_agument](image)
        else:
            image_transform = self.transform(image)

        return image_transform, multi_class_label
    
    def split_dataset(self) -> Tuple[Subset, Subset]:
        min_cnt = Counter(self.class_labels).most_common()[-1][1]
        n_per_class = int(min_cnt * .2)
        val_indices = set()
        for i in range(18):
            val_indices = val_indices|set(random.sample(self.class_dict[i], k=n_per_class))
        train_indices = set(range(len(self.class_labels))) - val_indices
        train_set, val_set = Subset(self, list(train_indices)), Subset(self, list(val_indices))
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌 사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다.
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        """프로필을 학습과 검증용으로 나누는 메서드"""
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
class AgeRateProfileDataset(MaskBaseDataset):

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        young = []
        middle = []
        old = []

        for i, profile in enumerate(profiles):
            age = int(profile.split('_')[-1])
            if age < 30:
                young.append(i)
            elif age < 60:
                middle.append(i)
            else:
                old.append(i)

        y_indices = set(random.sample(young, k=256))
        m_indices = set(random.sample(middle, k=245))
        o_indices = set(random.sample(old, k=39))

        val_indices = y_indices|m_indices|o_indices
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
class GenderAgeRateProfileDataset(MaskBaseDataset):

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        young_female = []
        young_male = []
        middle_female = []
        middle_male = []
        old_female = []
        old_male = []
        name = [young_female, young_male, middle_female, middle_male, old_female, old_male]
        for i, profile in enumerate(profiles):
            id, gender, race, age = profile.split('_')
            if int(age) < 30:
                if gender == 'male':
                    young_male.append(i)
                else:
                    young_female.append(i)
            elif int(age) < 60:
                if gender == 'male':
                    middle_male.append(i)
                else:
                    middle_female.append(i)
            else:
                if gender == 'male':
                    old_male.append(i)
                else:
                    old_female.append(i)

        yf = set(random.sample(young_female, k=int(256*.61)))
        ym = set(random.sample(young_male, k=int(256*.39)+1))
        mf = set(random.sample(middle_female, k=int(245*.61)))
        mm = set(random.sample(middle_male, k=int(245*.39)+1))
        of = set(random.sample(old_female, k=int(39*.61)))
        om = set(random.sample(old_male, k=int(39*.39)+1))

        val_indices = yf|ym|mf|mm|of|om
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    """테스트 데이터셋 클래스"""

    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.img_paths)


class MaskMultiLabelDataset(MaskBaseDataset):
    num_classes = 3 + 2 + 3

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)

        image_transform = self.transform(image)
        return image_transform, mask_label, gender_label, age_label


class MultiLabelSplitByProfileDataset(MaskBaseDataset):
    num_classes = 3 + 2 + 3

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        """프로필을 학습과 검증용으로 나누는 메서드"""
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)

        image_transform = self.transform(image)
        return image_transform, mask_label, gender_label, age_label

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