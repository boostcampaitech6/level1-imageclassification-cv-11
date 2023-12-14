import json
import time
import redis
import env
import mysql_query
from utils import start_subprocess, check_installation

def consume_messages():
    # Redis 연결 설정
    redis_client = redis.Redis(host=env.PUBLISH_IP, port=env.PUBLISH_REDIS_PORT, db=0)

    # python module check
    check_installation('git')
    import git

    # git 인증 캐싱
    start_subprocess('git config credential.helper cache')
    start_subprocess('git config credential.helper \'cache --timeout=1036800\'')
    start_subprocess('git pull')

    args_name = ['id', 'branch', 'name', 'seed', 'epoch', 'dataset', 'augmentation',
                 'resize', 'batch_size', 'valid_batch_size', 'model', 'optimizer',
                 'lr', 'val_ratio', 'criterion', 'lr_decay_step', 'log_interval',
                 'patience', 'data_dir', 'model_dir'
                 ]

    # consumer 작업 실행
    print('consumer 작업 실행')
    while True:

        # redis-server에 연결
        pubsub = redis_client.pubsub()

        # 큐 구독
        pubsub.subscribe(env.QUEUE_NAME)

        # 메시지 pop
        message = pubsub.get_message()
        if message['type'] == 'message':
            message_id = json.loads(message['data'])

            #0:id, 1:branch, 2:name, 3:seed, 4:epoch, 5:dataset, 6:augmentation
            #7:resize, 8:batch_size, 9:valid_batch_size, 10:model, 11:optimizer
            #12:lr, 13:val_ratio, 14:criterion, 15:lr_decay_step, 16:log_interval
            #17:patience, 18:data_dir, 19:model_dir
            args_value = mysql_query.select_consumer(message_id)

            branch = args_value[1]
            repo = git.Repo.init(path=env.CWD)
            repo.remotes.origin.pull()
            repo.git.checkout(branch)
            # repo.remotes.origin.pull()
            print(f'checkout {branch}')

            args = ''
            for idx, name in enumerate(args_name):
                args += f'--{name} {args_value[idx]} '

            train = start_subprocess(f'python {env.CWD}/train.py {args}')
            if train.stderr:
                mysql_query.insert_error_log(message_id, train.stderr)

        time.sleep(10)
    

if __name__ == "__main__":
    consume_messages()