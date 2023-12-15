import argparse
from utils import check_installation

check_installation(kind='redis')
check_installation(kind='pymysql')
    
import publisher
import consumer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=int, default=0, help="publisher = 1, consumer = 0"
)
args = parser.parse_args()

if args.mode:
    publisher.publish_tasks()
else:
    consumer.consume_messages()