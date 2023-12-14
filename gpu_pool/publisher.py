import time
import redis
import env
import json
import mysql_query
from utils import start_server

redis_client = redis.Redis(host=env.PUBLISH_IP, port=env.PUBLISH_REDIS_PORT, db=0)

def publish_tasks():
    # Redis 연결 설정

    # database server 시작
    start_server(kind='redis')
    start_server(kind='mysql')

    # mysql 데이터베이스, 테이블 생성 (해당 데이터베이스와 테이블이 없는 경우만)
    mysql_query.create_database()
    mysql_query.create_table()   

    # publish 작업 실행
    print("publisher 작업 실행")
    while True:
        print('작업 새로고침')
        # mysql db에서 새로 업로드 된 메시지들을 확인
        for message in mysql_query.select_publisher():

            # message의 id 추출
            message_id = message[0]

            # mysql에서 message의 pushed를 1로 변경
            # mysql_query.update_message(message_id)

            # redis 큐에 message_id 추가
            redis_client.publish(env.QUEUE_NAME, message_id)

        time.sleep(60)

if __name__ == "__main__":
    publish_tasks()