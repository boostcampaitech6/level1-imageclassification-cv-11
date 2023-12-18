import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class ResNetBasedModel(nn.Module):
    """
    ResNet을 기반으로 하는 프리트레인된 모델을 사용한 성별, 나이, 마스크 착용 예측 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어를 초기화합니다.

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        # ResNet 모델을 불러옵니다. 여기에서는 ResNet50을 예로 들겠습니다.
        # pretrained=True는 사전 트레이닝된 가중치를 사용하겠다는 의미입니다.
        self.base_model = models.resnet50(pretrained=True)

        # ResNet의 마지막 완전연결 레이어를 교체합니다.
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # 입력 데이터를 모델에 통과시킵니다.
        x = self.base_model(x)
        return x



class CustomInceptionV3(nn.Module):
    """
    참고사항: 이미지 사이즈 299 x 299 적당함
    """
    def __init__(self, num_classes):
        super(CustomInceptionV3, self).__init__()
        # 사전 훈련된 InceptionV3 모델 불러오기
        self.model = timm.create_model('inception_v3', pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False

        params = list(self.model.parameters())

        for param in params[-3:]:  #맨 뒤 3개
            param.requires_grad = True
        

        # 모델의 마지막 레이어 교체
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        return self.model(x)