# CV-11 기초대회 베이스라인 코드

## Project Structure

```
${PROJECT}
├── gpu_pool
├── data
|   ├── eval
|   ├── train
├── model
├── output
├── loss.py
├── model.py
├── dataset.py
├── train.py
├── inference.py
├── README.md
└── requirements.txt
```

- dataset.py : This file contains dataset class for model training and validation
- inference.py : This file used for predict the model
- loss.py : This file defines the loss functions used during training
- model.py : This file defines the model
- README.md
- requirements.txt : contains the necessary packages to be installed
- train.py : This file used for training the model

## Getting Started

### Install Requirements

To Insall the necessary packages liksted in `requirements.txt`, run the following command while your virtual environment is activated:


```
pip install -r requirements.txt
```

### Usage

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.

```
python train.py
```

#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.

```
python inference.py --model BaseModel --model_dir ./model/exp
```

#### Tensorboard
```
tensorboard --logdir=save_dir
```

#### WandB
```
pip install wandb
wandb login
```

- - -

# GPU pool

## Getting Started

### Requirements

Publisher인 경우:
- MySQL Server (MySQL >= 5.7)
  - env.py에 입력한 user -> MySQL Server에서 데이터베이스 접근, 생성 권한을 부여하고 외부 접속을 허용해야 함.

### Usage

- **Publisher와 Consumer는 사용 전에 env.py 수정이 필요합니다.**

#### Publisher

```
# foreground
python gpu_pool --mode 1

# background
nohup python gpu_pool --mode 1 &
```

#### Consumer
- 첫 실행은 foreground로 하여 반복되는 git checkout 작업을 위해 git 아이디와 액세스 토큰을 입력해야 합니다. 아이디와 액세스 토큰은 11일 동안 캐싱 됩니다. 이 기간 동안 background 실행이 가능합니다.
- 11일이 지난 이후엔 다시 foreground로 실행하여 아이디와 액세스 토큰을 입력해야 합니다.
```
# foreground
python gpu_pool

# background
nohup python gpu_pool &
```

#### Insert message in queue

```
python gpu_pool/push.py
```

### Example

#### env

```
# publisher의 ip와 port
PUBLISH_IP = '10.28.xxx.xx'
PUBLISH_REDIS_PORT = 300066
PUBLISH_MYSQL_PORT = 300067
PUBLISH_MYSQL_USER = 'user'
PUBLISH_MYSQL_PASSWORD = 'password'

# 큐 이름
QUEUE_NAME = 'ai_train_queue'

# 큐에 푸시하려는 메시지 파일 주소
MESSAGE_FILE_DIR = './message.json'
```

#### message

```
# ./message.json
{
  "camper_id": "name",
  "branch": "develop",   # or commit ex) e6084c6b
  "name": "test_name",
  "seed": "42",
  "epoch": "1",
  "dataset": "MaskBaseDataset",
  "augmentation": "BaseAugmentation",
  "resize": "128 96",
  "batch_size": "64",
  "valid_batch_size": "1000",
  "model": "BaseModel",
  "optimizer": "SGD",
  "lr": "1e-3",
  "val_ratio": "0.2",
  "criterion": "cross_entropy",
  "lr_decay_step": "20",
  "log_interval": "20",
  "patience": "5",
  "data_dir": "test_data_dir",
  "model_dir": "test_model_dir"
}
```
