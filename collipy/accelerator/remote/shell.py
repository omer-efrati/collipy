"""
Remote server shell access manager
"""
import paramiko


class ShellHandler:
    """

    """
    def __init__(self, host: str, port: int, user: str, password: str):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print('Connecting host...')
        try:
            self.ssh.connect(hostname=host, port=port, username=user, password=password)
        except paramiko.SSHException:
            raise paramiko.SSHException('Cannot login to server\nPlease try again...')
        print('Connection established')
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
