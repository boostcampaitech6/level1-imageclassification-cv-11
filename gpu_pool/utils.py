import subprocess
import env

def start_subprocess(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, shell=True)

def start_server(kind: str) -> None:
    if kind not in ['redis', 'mysql']:
        raise Exception('잘못된 입력')
    
    kind_fn = 'redis-server' if kind == 'redis' else 'mysql-server' # redis or mysql

    # server 설치 확인
    check_installation(kind=kind_fn)

    # 설정 파일 확인
    check_config(kind=kind)

    if kind == 'redis':
        # 설정 파일 복사/붙여넣기
        start_subprocess(f'cp {env.CWD}/gpu_pool/cnf/redis.conf /etc/redis.conf')
    else:
        # 설정 파일 복사/붙여넣기
        start_subprocess(f'cp {env.CWD}/gpu_pool/cnf/mysqld.cnf /etc/mysql/mysql.conf.d/mysqld.cnf')

    # server 서버 실행
    print(f'{kind_fn} 실행 중...')
    start = start_subprocess('redis-server /etc/redis.conf') if kind == 'redis' else start_subprocess('service mysql start')
    if start.stderr:
        print (f'{kind_fn}를 실행하지 못했습니다. 터미널 로그는 다음과 같습니다. \n', start.stderr)
        raise Exception(f'{kind_fn} 실행 실패')

def check_config(kind: str) -> None:
    if kind == 'redis':
        config_dir = f'{env.CWD}/gpu_pool/cnf/redis.conf'
        port = f'\nport {env.PUBLISH_REDIS_PORT}'
    else:
        config_dir = f'{env.CWD}/gpu_pool/cnf/mysqld.cnf'
        port = f'\nport = {env.PUBLISH_MYSQL_PORT}'

    with open(config_dir, 'r') as file:
        if file.readlines()[-1] == '##':
            with open(config_dir, 'a') as file:
                file.write(port)


def check_installation(kind: str) -> None:
    if kind == 'redis': 
        command_c = 'pip3 show redis'
        command_i = 'pip3 install redis'
    elif kind == 'redis-server':
        command_c = 'redis-server --version'
        command_i = 'apt-get install -y redis-server'
    elif kind == 'pymysql':
        command_c = 'pip3 show pymysql'
        command_i = 'pip3 install pymysql'
    elif kind == 'mysql-server':
        command_c = 'mysql --version'
    elif kind == 'git':
        command_c = 'pip3 show gitpython'
        command_i = 'pip3 install gitpython'
    else:
        raise Exception(f'지정오류')
    
    # 확인 파트
    print(f'{kind} 설치 확인 중...')
    check = start_subprocess(command_c)
    if check.stderr:
        print(f'{kind} 설치되어 있지 않습니다. 설치를 시작합니다.')

        if kind == 'redis-server':
            start_subprocess('apt-get update && apt-get install -y --no-install-recommends')
            start_subprocess('apt-get install -y dialog apt-utils systemctl')
        elif kind == 'mysql-server':
            print(f'{kind} 는 따로 설치가 필요합니다. 설치 후 user=root/password=\'\'로 외부접속허용 설정이 요구됩니다.')
            raise Exception(f'{kind} 설치되지 않음')

        # 설치 파트
        print(f'{kind} 설치 중...')
        install = start_subprocess(command_i)
        if install.stderr and 'WARNING: Running pip as the \'root\' user can result in broken permissions' not in install.stderr:
            print(f'{kind} 설치에 실패했습니다. 터미널 로그는 다음과 같습니다. \n', install.stderr)
            raise Exception(f'{kind} 설치 실패')