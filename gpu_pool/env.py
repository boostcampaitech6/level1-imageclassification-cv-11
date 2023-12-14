import os

# 프로젝트 디렉토리 주소
CWD = os.getcwd()

# publisher의 ip와 port  
PUBLISH_IP = 'ip_address' # ex)'10.28.224.65'
PUBLISH_REDIS_PORT = 0 # ex) 30066
PUBLISH_MYSQL_PORT = 0 # ex) 30067

# redis 큐 이름
QUEUE_NAME = 'ai_train_queue'

# redis 큐에 푸시하려는 메시지 파일 주소
MESSAGE_FILE_DIR = 'file_dir' # ex) './message.json' or f'{CWD}/message.json'

# message.json 양식 예
# {
#   "branch": "develop",   // or commit ex) e6084c6b
#   "name": "철수",
#   "seed": "42",
#   "epoch": "1",
#   "dataset": "MaskBaseDataset",
#   "augmentation": "BaseAugmentation",
#   "resize": "128 96",
#   "batch_size": "64",
#   "valid_batch_size": "1000",
#   "model": "BaseModel",
#   "optimizer": "SGD",
#   "lr": "1e-3",
#   "val_ratio": "0.2",
#   "criterion": "cross_entropy",
#   "lr_decay_step": "20",
#   "log_interval": "20",
#   "patience": "5",
#   "data_dir": "test_data_dir",
#   "model_dir": "test_model_dir"
# }