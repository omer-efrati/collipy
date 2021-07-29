"""
SSH Communications
==================

Remote secured shell access manager

Notes
-----
This module supports threading

"""
import re
import threading
import paramiko
from .geantparser import parse


class ShellHandler:
    """
    Basic interactive shell handler
    """
    def __init__(self, host: str, port: int, username: str, password: str):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print('Connecting host...')
        self.ssh.connect(hostname=host, port=port, username=username, password=password)
        print('Connection established')
        self.channel = self.ssh.invoke_shell()
        self.stdin = self.channel.makefile('wb')
        self.stdout = self.channel.makefile('r')
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ssh.close()

    def __del__(self):
        self.ssh.close()

    def execute(self, cmd: str, cmd_finish: str, prompt_finish: str) -> str:
        """Executing command"""
        cmd = cmd.strip('\n')
        cmd_finish = cmd_finish.strip('\n')
        prompt_finish = prompt_finish.strip('\r\n')
        prompt_finish += '\r\n'
        # if using threading, prevents concurrently sending commands through the same connection
        with self.lock:
            self.stdin.write(f'{cmd}\n{cmd_finish}\n')
            out = ''
            for line in self.stdout:
                if line == prompt_finish:
                    break
                out += line
        return out


class GSH(ShellHandler):
    """Geant Shell Handler"""

    def __init__(self, username: str, password: str, alpha: float):
        host = "hpcssd.tau.ac.il"
        port = 22
        super().__init__(host, port, username, password)
        print('Initializing simulator')
        self._start_gimel(alpha)
        print('Simulator is ready!')

    def _start_gimel(self, alpha):
        """Initializing gimel program"""
        self.execute('cd /var/misc/phys', 'echo odeo47', 'odeo47')
        self.execute('singularity shell --bind /var/misc/phys /docker_scratch/g', 'echo odeo47', 'odeo47')
        self.stdin.write('./gimel\n')
        cnt = 0
        out = ''
        for line in self.stdout:
            out += line
            match = re.fullmatch(r'\s*\*+\s*', line)
            if match:
                cnt += 1
            if cnt == 4:
                break
        cmd = '0\n'
        cmd += 2 * 'Y\n' + f'{alpha}\n' + 3 * 'N\n' if alpha != 1 else 'N\n'
        self.execute(cmd, 'odeo47', ' *** Unknown command: odeo47')

    def inject(self, particle: str, momentum: float, times: int):
        cmd = f'{particle} {momentum}\n' + times * 'inject\n'
        txt = self.execute(cmd, 'odeo47', ' *** Unknown command: odeo47\r\n')
        return parse(txt)
