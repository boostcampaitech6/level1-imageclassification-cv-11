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

class CustomEfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetB3, self).__init__()
        # 사전 훈련된 EfficientNet-B3 모델 불러오기
        self.model = timm.create_model('efficientnet_b3', pretrained=True)

        # 모델의 초기 층의 가중치를 고정 (그라디언트 계산 비활성화)
        for param in self.model.parameters():
            param.requires_grad = False

        # 모델의 후반부 층의 가중치를 미세 조정 (그라디언트 계산 활성화)
        # 예시로, 마지막 블록(block)과 분류 레이어(classifier)에 대해 그라디언트를 활성화
        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # 마지막 분류 레이어 교체
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        # 사전 훈련된 ResNet-50 모델 불러오기
        self.model = models.resnet50(pretrained=True)

        # 모델의 모든 파라미터에 대해 그라디언트 계산 비활성화
        for param in self.model.parameters():
            param.requires_grad = False

        # layer4의 마지막 세 개의 Bottleneck 레이어에 대해 그라디언트 계산 활성화
        for layer in self.model.layer4[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 모델의 마지막 FC 레이어 교체 및 그라디언트 계산 활성화
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(CustomInceptionV3, self).__init__()
        # 사전 훈련된 Inception V3 모델 불러오기
        self.model = models.inception_v3(pretrained=True)

        # 모든 레이어의 그라디언트 업데이트 비활성화
        for param in self.model.parameters():
            param.requires_grad = False

        # 마지막 3개 레이어의 그라디언트 업데이트 활성화
        for layer in [self.model.Mixed_7b, self.model.Mixed_7c, self.model.fc]:
            for param in layer.parameters():
                param.requires_grad = True

        # 마지막 FC 레이어를 주어진 클래스 수에 맞게 교체
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 보조 출력 레이어를 주어진 클래스 수에 맞게 교체 (필요한 경우)
        if self.model.AuxLogits is not None:
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)

    def forward(self, x):
        if self.training:
            main_output, aux_output = self.model(x)
            return main_output + 0.3 * aux_output
        else:
            # 평가 모드에서는 주 출력만 반환
            main_output = self.model(x)
            return main_output


class CustomEfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetB4, self).__init__()
        # 사전 훈련된 EfficientNet-B4 모델 불러오기
        self.model = timm.create_model('efficientnet_b4', pretrained=True)

        # 모든 레이어의 그라디언트 업데이트 비활성화
        for param in self.model.parameters():
            param.requires_grad = False

        # 마지막 블록과 분류 레이어의 그라디언트 업데이트 활성화
        # EfficientNet의 경우 최종 블록을 정확히 지정하는 것이 중요합니다.
        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        # 사전 훈련된 ViT 모델 불러오기
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)

        # 모든 레이어의 그라디언트 업데이트 비활성화
        for param in self.model.parameters():
            param.requires_grad = False

        # ViT의 마지막 분류 레이어를 주어진 클래스 수에 맞게 교체
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class CustomInceptionV3_NOAUX(nn.Module):
    def __init__(self, num_classes):
        super(CustomInceptionV3, self).__init__()
        # 사전 훈련된 Inception V3 모델 불러오기
        self.model = models.inception_v3(pretrained=True)

        # 모든 레이어의 그라디언트 업데이트 비활성화
        for param in self.model.parameters():
            param.requires_grad = False

        # 마지막 3개 레이어의 그라디언트 업데이트 활성화
        for layer in [self.model.Mixed_7b, self.model.Mixed_7c, self.model.fc]:
            for param in layer.parameters():
                param.requires_grad = True

        # 마지막 FC 레이어를 주어진 클래스 수에 맞게 교체
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
    # Inception V3 모델의 출력이 (주 출력, 보조 출력) 튜플 형태인 경우와 그렇지 않은 경우를 처리
        outputs = self.model(x)
    # 훈련 모드에서는 주 출력과 보조 출력이 함께 반환되지만, 여기서는 주 출력만 사용
        if self.training:
            main_output, _ = outputs
        else:
            main_output = outputs
        return main_output