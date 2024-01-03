import json
import time
import redis
import env
import mysql_query
from utils import start_subprocess, check_installation

redis_client = redis.Redis(host=env.PUBLISH_IP, port=env.PUBLISH_REDIS_PORT, db=0)

def consume_messages():
    check_installation('git')
    import git

    start_subprocess('git config credential.helper cache')
    start_subprocess('git config credential.helper \'cache --timeout=1036800\'')
    start_subprocess('git pull')

    message_columns = ['id', 'camper_id', 'branch', 'name', 'seed', 'epoch', 'dataset', 'augmentation',
                 'resize', 'batch_size', 'valid_batch_size', 'model', 'optimizer',
                 'lr', 'val_ratio', 'criterion', 'lr_decay_step', 'log_interval',
                 'patience', 'data_dir', 'model_dir'
                 ]
    args_exception = ['id', 'branch']
    branch_index = 2

    print('consumer 작업 실행')
    while True:
        print('작업 새로고침')
        pubsub = redis_client.pubsub()
        pubsub.subscribe(env.QUEUE_NAME)

        for message in pubsub.listen():
            if message and message['type'] =='subscribe':
                continue

            message_id = 0

            try:
                if message and message['type'] == 'message':
                    message_id = json.loads(message['data'])
                    if mysql_query.update_message(message_id):
                        args_value = mysql_query.select_consumer(message_id)

                        branch = args_value[branch_index]
                        repo = git.Repo.init(path=env.CWD)
                        repo.remotes.origin.pull()
                        repo.git.checkout(branch)
                        repo.remotes.origin.pull()
                        print(f'checkout {branch}')

                        args = ''
                        for idx, name in enumerate(message_columns):
                            if name in args_exception:
                                continue
                            args += f'--{name} {args_value[idx]} '

                        print('학습 중')
                        train = start_subprocess(f'python {env.CWD}/train.py {args}')
                        if train.stderr and 'UserWarning: Argument \'interpolation\' of type int is deprecated since 0.13 and will be removed in 0.15' not in train.stderr:
                            mysql_query.insert_error_log(message_id, train.stderr)
                            print(f'학습 중 에러 발생. error_message에서 {message_id}를 참조바랍니다.')
                        else:
                            print('학습 완료')
            except Exception as e:
                mysql_query.insert_error_log(message_id, e)
                print(f'에러 발생. error_message에서 {message_id}를 참조바랍니다.')
                
                    
            break

        time.sleep(60)
    

if __name__ == "__main__":
    consume_messages()