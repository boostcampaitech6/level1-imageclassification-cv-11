import pymysql 
import env

# 데이터베이스 생성
def create_database():
    conn_no_db = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD, charset='utf8') 
    sql = 'create database IF NOT EXISTS ai_train;'
    with conn_no_db:
        with conn_no_db.cursor() as cur:
            cur.execute(sql)

# 테이블 생성
def create_table():
    conn = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD,db="ai_train", charset='utf8') 
    sqls = [
        '''create table IF NOT EXISTS message(
            id int(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
            branch varchar(255),
            name varchar(255),
            seed varchar(255),
            epoch varchar(255),
            dataset varchar(255),
            augmentation varchar(255),
            resize varchar(255),
            batch_size varchar(255),
            valid_batch_size varchar(255),
            model varchar(255),
            optimizer varchar(255),
            lr varchar(255),
            val_ratio varchar(255),
            criterion varchar(255),
            lr_decay_step varchar(255),
            log_interval varchar(255),
            patience varchar(255),
            data_dir varchar(255),
            model_dir varchar(255),
            pushed bool DEFAULT FALSE
        );''',
        '''create table IF NOT EXISTS error_message(
            id int(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
            message_id int(10) NOT NULL,
            error_log LONGTEXT
        );'''
        ]
    with conn:
        with conn.cursor() as cur:
            for sql in sqls:
                cur.execute(sql)

# publisher에게 message 중 pushed가 0인 row들을 반환
def select_publisher() -> list[tuple[any]]:
    conn = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD,db="ai_train", charset='utf8') 
    sql = 'SELECT id FROM message WHERE pushed != 1'
    result = []
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            result = cur.fetchall()   
    return result
    
# consumer에게 message 중 id와 message_id가 일치하는 row를 반환
def select_consumer(message_id: int) -> tuple[any]:
    conn = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD,db="ai_train", charset='utf8') 
    sql = f'SELECT * FROM message WHERE id = {message_id}'
    result = tuple()
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            result = cur.fetchone()   
    return result

# 서버에 메시지 푸시
def insert_message(json: dict) -> None:
    conn = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD,db="ai_train", charset='utf8') 
    keys = f'{tuple(json.keys())}'.replace('\'', '')
    values = tuple(json.values())

    sql = f'INSERT message {keys} VALUES {values}'
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

# publisher가 redis 큐에 담은 message의 message_id와 일치하는 row의 pushed를 1로 변경
def update_message(message_id: int) -> None:
    conn = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD,db="ai_train", charset='utf8') 

    sql = f'UPDATE message SET pushed = 1 WHERE id = {message_id}'
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

# consumer가 train에 실패하면 해당 에러 로드와 message_id를 mysql db에 저장
def insert_error_log(message_id:int, error_log:str) -> None:
    conn = pymysql.connect(host=env.PUBLISH_IP,port=env.PUBLISH_MYSQL_PORT, user=env.PUBLISH_MYSQL_USER, password=env.PUBLISH_MYSQL_PASSWORD,db="ai_train", charset='utf8') 

    sql = 'INSERT error_message (message_id, error_log) VALUES (%s, %s)'
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, (message_id, error_log))
        conn.commit()