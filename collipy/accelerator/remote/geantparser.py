"""
Geant Parser
============

Parsing text from geant shell

"""
import pandas as pd


# Headline for injections properties
ECAL = '          ELECTROMAGNETIC CLUSTERS'
TRACK = '          CHARGED TRACKS RECONSTRUCTION'
AKAPPA = 'AKAPPA'
VERTEX = '          CHARGED TRACKS VERTECES RECONSTRUCTION'
VERTEXES = '  Final coordinates of the vertex:'


def _injection_parse(injection: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parsing a single injection

    Notes
    =====
    ValueError and IndexError will be raised in the try block for one of the following reasons:

    * GEANT fails to analyze a measurement (gives uncertainty: "**********")

    * the (necessary) finish command might fall in a place where it will interrupt the parsing process

    When one of these cases occur the injections will be discarded. These cases are extremely rare therefore
    their effect on statistics and such is insignificant.

    """
    vertex = pd.DataFrame(columns=['phi', 'dphi', 'x', 'dx', 'y', 'dy', 'z', 'dz'])
    track = pd.DataFrame(columns=['k', 'dk', 'tan_theta', 'dtan_theta'])
    ecal = pd.DataFrame(columns=['ph', 'x', 'dx', 'y', 'dy', 'z', 'dz'])
    if VERTEX in injection:
        n = [i for i in range(len(injection)) if VERTEXES in injection[i]]
        for i in n:
            try:
                row = [float(injection[i + 5].split()[1]), float(injection[i + 5].split()[3]),
                       float(injection[i + 1].split()[2]) + 10, float(injection[i + 1].split()[4]),
                       # x_0 is at -10cm at geant
                       float(injection[i + 2].split()[2]), float(injection[i + 2].split()[4]),
                       float(injection[i + 3].split()[2]), float(injection[i + 3].split()[4])]
                vertex = vertex.append(pd.Series(row, index=vertex.columns), ignore_index=True)
            except (ValueError, IndexError):
                return None
    if TRACK in injection:
        n = [i for i in range(len(injection)) if AKAPPA in injection[i]]
        # For each track there are 3 `AKAPPA` prints. Next line chooses the first of which.
        n = [n[i] for i in range(0, len(n), 3)]
        for i in n:
            try:
                row = [float(injection[i].split()[1]), float(injection[i + 10].split()[2]),
                       float(injection[i + 3].split()[1]), float(injection[i + 3 + 10].split()[5])]
                track = track.append(pd.Series(row, index=track.columns), ignore_index=True)
            except (ValueError, IndexError):
                return None
    if ECAL in injection:
        i = injection.index(ECAL) + 3
        row = injection[i].split()
        while row[0].isdigit():
            try:
                row = [float(t.replace('+/-', '')) for t in row][1:8]
                row[1] += 10  # x_0 is at -10cm at the simulator
                ecal = ecal.append(pd.Series(row, index=ecal.columns), ignore_index=True)
            except (ValueError, IndexError):
                return None
            i += 1
            if i < len(injection):  # last row in inject might be an EM cluster at the end of an injection
                row = injection[i].split()
            else:
                break

    return vertex, track, ecal


def parse(txt: str) -> [(pd.DataFrame, pd.DataFrame, pd.DataFrame)]:
    """Parsing injections data from GEANT software

    Parameters
    ----------
        txt
            text received from GEANT after one or multiple injections

    Returns
    -------
        res
            list of tuples, each consists of the following data: (vertexes, tracks, ecal indications)
    """
    res = []
    # each injection is bounded by this row
    injections = txt.split('GEANT > inject\r\n')
    injections.pop(0)     # information about the particle and momentum is stored in this row
    for _, inject in enumerate(injections):
        # remove all blank spaces
        inject = inject.split('\r\n')
        inject = list(filter(None, inject))
        # parsing a single injection
        inj = _injection_parse(inject)
        if inj is not None:
            res.append(inj)
    return res
