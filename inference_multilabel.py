import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

classes = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 0, 2],
    3: [0, 1, 0],
    4: [0, 1, 1],
    5: [0, 1, 2],
    6: [1, 0, 0],
    7: [1, 0, 1],
    8: [1, 0, 2],
    9: [1, 1, 0],
    10: [1, 1, 1],
    11: [1, 1, 2],
    12: [2, 0, 0],
    13: [2, 0, 1],
    14: [2, 0, 2],
    15: [2, 1, 0],
    16: [2, 1, 1],
    17: [2, 1, 2],
}


def load_model(saved_model, num_classes, device):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    모델 추론을 수행하는 함수

    Args:
        data_dir (str): 테스트 데이터가 있는 디렉토리 경로
        model_dir (str): 모델 가중치가 저장된 디렉토리 경로
        output_dir (str): 결과 CSV를 저장할 디렉토리 경로
        args (argparse.Namespace): 커맨드 라인 인자

    Returns:
        None
    """

    # CUDA를 사용할 수 있는지 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 클래스의 개수를 설정한다. (마스크, 성별, 나이의 조합으로 18)
    # num_classes = MaskBaseDataset.num_classes  # 18
    num_classes = 8
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            mask_outs, gender_outs, age_outs = torch.split(pred, [3, 2, 3], dim=1)
            mask_preds, gender_preds, age_preds = (
                    torch.argmax(mask_outs, dim=-1),
                    torch.argmax(gender_outs, dim=-1),
                    torch.argmax(age_outs, dim=-1),
            )
            pred = torch.Tensor(
                [
                    [
                        k
                        for k, v in classes.items()
                        if v == [mask_preds[i], gender_preds[i], age_preds[i]]
                    ]
                    for i in range(args.batch_size)
                ]
            )
            pred = pred.squeeze().type(torch.int64)
            preds.extend(pred.cpu().numpy())

    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(96, 128),
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
    )

    args = parser.parse_args()

    data_dir = "./data/eval"
    model_dir = f"./model/{args.model_dir}"
    output_dir = f"./output/{args.model_dir}"

    os.makedirs(output_dir, exist_ok=True)

    # 모델 추론을 수행한다.
    inference(data_dir, model_dir, output_dir, args)
