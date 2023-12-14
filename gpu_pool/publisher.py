import time
import redis
import env
import json
import mysql_query
from utils import start_server

MESSAGE_ID_INDEX = 0

redis_client = redis.Redis(host=env.PUBLISH_IP, port=env.PUBLISH_REDIS_PORT, db=0)

def publish_tasks():
    start_server(kind='redis')
    start_server(kind='mysql')

    mysql_query.create_database()
    mysql_query.create_table()   

    print("publisher 작업 실행")
    while True:
        print('작업 새로고침')
        for message in mysql_query.select_publisher():
            message_id = message[MESSAGE_ID_INDEX]
            redis_client.publish(env.QUEUE_NAME, message_id)
        time.sleep(60)

if __name__ == "__main__":
    publish_tasks()