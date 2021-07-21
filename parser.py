import pandas as pd


ECAL = '          ELECTROMAGNETIC CLUSTERS'
TRACK = '          CHARGED TRACKS RECONSTRUCTION'
AKAPPA = 'AKAPPA'
VERTEX = '          CHARGED TRACKS VERTECES RECONSTRUCTION'
VERTEXES = '  Final coordinates of the vertex:'


def _injection_parse(injection: str) -> list[(pd.DataFrame, pd.DataFrame, pd.DataFrame)]:
    """Parsing a single injection text
    Notes
    =====
    discarding measurements GEANT fails to analyze (gives uncertainty: "**********")
    """
    if VERTEX in injection:
        n = [i for i in range(len(injection)) if VERTEXES in injection[i]]
        for i in n:
            try:
                row = [float(injection[i + 5].split()[1]), float(injection[i + 5].split()[3]),
                       float(injection[i + 1].split()[2]) + 10, float(injection[i + 1].split()[4]),
                       # x_0 is at -10cm at geant
                       float(injection[i + 2].split()[2]), float(injection[i + 2].split()[4]),
                       float(injection[i + 3].split()[2]), float(injection[i + 3].split()[4])]
                vertex = pd.DataFrame(row, columns=['phi', 'dphi', 'x', 'dx', 'y', 'dy', 'z', 'dz'])
            except ValueError:
                return [], [], []
    if TRACK in injection:
        n = [i for i in range(len(injection)) if AKAPPA in injection[i]]
        n = [n[i] for i in range(0, len(n), 3)]
        for i in n:
            try:
                row = [float(injection[i].split()[1]), float(injection[i + 10].split()[2]),
                       float(injection[i + 3].split()[1]), float(injection[i + 3 + 10].split()[5])]
                track = pd.DataFrame(row, columns=['k', 'dk', 'tan_theta', 'dtan_theta'])
            except ValueError:
                return [], [], []
    if ECAL in injection:
        i = injection.index(ECAL) + 3
        row = injection[i].split()
        while row[0].isdigit():
            try:
                row = [float(t.replace('+/-', '')) for t in row][1:8]
                row[1] += 10  # x_0 is at -10cm at the simulator
                ecal = pd.DataFrame(row, columns=['ph', 'x', 'dx', 'y', 'dy', 'z', 'dz'])

            except ValueError:
                return [], [], []
            i += 1
            if i < len(injection):  # last row in inject might be an EM cluster at the end of an injection
                row = injection[i].split()
            else:
                break
    return vertex, track, ecal


def parse(txt: str) -> list:
    """Parsing data from GEANT software"""
    res = []
    injections = txt.split('GEANT > inject\r\n')
    injections.pop(0)
    for _, inject in enumerate(injections):
        inject = inject.split('\r\n')
        inject = list(filter(None, inject))
        res.append(_injection_parse(inject))
    return res