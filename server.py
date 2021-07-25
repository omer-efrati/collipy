import logging
import time
import paramiko
from geant_parser import parse


logger = logging.getLogger('serverLog')


class ShellHandler:
    def __init__(self, host: str, port: str, user: str, password: str):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logger.info('Connecting host...')
        try:
            self.ssh.connect(hostname=host, port=port, username=user, password=password)
        except paramiko.SSHException:
            raise paramiko.SSHException('Cannot login to server\nPlease try again...')
        logger.info('Connection established')
        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ssh.close()

    def __del__(self):
        self.ssh.close()

    def execute(self, cmd: str, finish: str, cmd_finish='') -> str:
        """Executing command"""
        cmd = cmd.strip('\n')
        self.stdin.write(f'{cmd}\n')
        if cmd_finish:
            cmd_finish = cmd_finish.strip('\n')
            self.stdin.write(f'{cmd_finish}\n')
        self.stdin.flush()
        out = ''
        for line in self.stdout:
            out += line
            if line == finish:
                break
        return out


class GSH(ShellHandler):
    """Geant Shell Handler"""

    def __init__(self, user: str, password: str, alpha: float):
        host = "hpcssd.tau.ac.il"
        port = 22
        super().__init__(host, port, user, password)
        logger.info('Initializing simulator')
        self._start_gimel(alpha)
        logger.info('Simulator is ready!')

    def _start_gimel(self, alpha):
        finish = 'END OF STDOUT\r\n'
        cmd_finish = f'echo {finish}'
        self.execute('cd /var/misc/phys', finish, cmd_finish)
        self.execute('singularity shell --bind /var/misc/phys /docker_scratch/g', finish, cmd_finish)
        time.sleep(1)
        self.execute('./gimel', ' Workstation type (?=HELP) <CR>=1 :   Invalid workstation type\r\n', 'odeo47')
        time.sleep(1)
        cmd = '0\n'
        cmd += 2 * 'Y\n' + f'{alpha}\n' + 3 * 'N\n' if alpha != 1 else 'N\n'
        self.execute(cmd, ' *** Unknown command: odeo47\r\n', 'odeo47')

    def inject(self, particle: str, momentum: float, times: int):
        cmd = f'{particle} {momentum}\n' + times * 'inject\n'
        txt = self.execute(cmd, ' *** Unknown command: odeo47\r\n', 'odeo47')
        return parse(txt)
