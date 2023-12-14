import json
import mysql_query
import env

with open(env.MESSAGE_FILE_DIR, 'r') as f:
    json_object = json.load(f)
    mysql_query.insert_message(json=json_object)
    print("메시지 푸시 완료")
