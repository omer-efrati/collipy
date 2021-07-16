import logging
import time
import paramiko
import pandas as pd


logger = logging.getLogger('colliderLog')


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
        logger.info(f'Initializing simulator')
        self._start_gimel(alpha)
        logger.info('Simulator is ready!')
        print()

    def _start_gimel(self, alpha):
        finish = 'END OF STDOUT\r\n'
        cmd_finish = f'echo {finish}'
        self.execute('cd /var/misc/phys', finish, cmd_finish)
        self.execute('singularity shell --bind /var/misc/phys /docker_scratch/g', finish, cmd_finish)
        self.execute('./gimel', ' Workstation type (?=HELP) <CR>=1 :   Invalid workstation type\r\n', 'odeo47')
        time.sleep(0.1)
        cmd = '0\n'
        cmd += 2 * 'Y\n' + f'{alpha}\n' + 3 * 'N\n' if alpha != 1 else 'N\n'
        self.execute(cmd, ' *** Unknown command: odeo47\r\n', 'odeo47')

    def inject(self, particle: str, momentum: float, times: int):
        cmd = f'{particle} {momentum}\n' + times * 'inject\n'
        txt = self.execute(cmd, ' *** Unknown command: odeo47\r\n', 'odeo47')
        return txt

    @staticmethod
    def parse(txt: str):
        """Parsing data from GEANT software"""
        ECAL = '          ELECTROMAGNETIC CLUSTERS'
        TRACK = '          CHARGED TRACKS RECONSTRUCTION'
        AKAPPA = 'AKAPPA'
        VERTEX = '          CHARGED TRACKS VERTECES RECONSTRUCTION'
        VERTEXES = '  Final coordinates of the vertex:'
        vindex = pd.MultiIndex.from_product([['vertex'], ['phi', 'dphi', 'x', 'dx', 'y', 'dy', 'z', 'dz']])
        tindex = pd.MultiIndex.from_product([['track'], ['k', 'dk', 'tan_theta', 'dtan_theta']])
        eindex = pd.MultiIndex.from_product([['ecal'], ['ph', 'dph', 'x', 'dx', 'y', 'dy', 'z', 'dz']])
        frames = []
        injections = txt.split('GEANT > inject\r\n')
        row = injections.pop(0).split()
        particle = row[2]   # TODO: DO YOU STILL NEED IT?
        momentum = float(row[3])
        for j, inject in enumerate(injections):
            row = []
            good = True  # discarding measurements GEANT fails to analyze (gives uncertainty: "**********")
            inject = inject.split('\r\n')
            inject = list(filter(None, inject))
            if VERTEX in inject:
                n = [i for i in range(len(inject)) if VERTEXES in inject[i]]
                for i in n:
                    try:
                        row = [float(inject[i + 5].split()[1]), float(inject[i + 5].split()[3]),
                               float(inject[i + 1].split()[2]) + 10, float(inject[i + 1].split()[4]),
                               # x_0 is at -10cm at geant
                               float(inject[i + 2].split()[2]), float(inject[i + 2].split()[4]),
                               float(inject[i + 3].split()[2]), float(inject[i + 3].split()[4])]
                        vertex = pd.DataFrame(row, columns=vindex)
                    except ValueError:
                        good = False
            if TRACK in inject:
                n = [i for i in range(len(inject)) if AKAPPA in inject[i]]
                n = [n[i] for i in range(0, len(n), 3)]
                for i in n:
                    try:
                        row = [float(inject[i].split()[1]), float(inject[i + 10].split()[2]),
                               float(inject[i + 3].split()[1]), float(inject[i + 3 + 10].split()[5])]
                        track = pd.DataFrame(row, columns=tindex)
                    except ValueError:
                        good = False
            if ECAL in inject:
                i = inject.index(ECAL) + 3
                row = inject[i].split()
                while row[0].isdigit():
                    try:
                        row = [float(t.replace('+/-', '')) for t in row][1:8]
                        row[1] += 10  # x_0 is at -10cm at the simulator
                        ecal = pd.DataFrame(row, columns=eindex)
                    except ValueError:
                        good = False
                    i += 1
                    if i < len(inject):  # last row in inject might be an EM cluster at the end of an injection
                        row = inject[i].split()
                    else:
                        break
            if good and len(row) != 0:
                frames.append(pd.concat([vertex, track, ecal], axis='columns'))
        return pd.concat(frames, keys=range(len(frames)), names=['injection'])
