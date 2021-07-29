"""
Injection Properties
"""

import numpy as np
import pandas as pd
from collections import namedtuple


class DecayMode:
    """
    Decay Mode property

    Parameters
    ----------
    vertex : int, DataFrame
        number of vertexes or vertex data frame
    track : int, DataFrame
        number of tracks or track data frame
    ecal : int, DataFrame
        number of ecal hits or ecal data frame

    Attributes
    ----------
    tup : tuple[float, float, float]
        (number of vertexes, number of tracks, number of ecal clusters)

    """

    Tup = namedtuple('DecayMode', ['nvertex', 'ntrack', 'necal'])

    def __init__(self, vertex, track, ecal):
        v = len(vertex) if type(vertex) == pd.DataFrame else vertex
        t = len(track) if type(track) == pd.DataFrame else track
        e = len(ecal) if type(ecal) == pd.DataFrame else ecal
        if type(v) == type(t) == type(e) == int:
            self.tup = DecayMode.Tup(v, t, e)
        else:
            raise ValueError("Expected arguments to be either int, DataFrame or None type")

    def __eq__(self, other):
        if self.tup == other.tup:
            return True
        return False

    def __le__(self, other):
        """Use <= as if 'self' mode contains 'other' mode"""
        if self.tup.nvertex is None or self.tup.nvertex == other.tup.nvertex:
            if self.tup.ntrack is None or self.tup.ntrack == other.tup.ntrack:
                if self.tup.necal is None or self.tup.necal == other.tup.necal:
                    return True
        return False

    def __ge__(self, other):
        return other.__le__(self)

    def __repr__(self):
        one = self.tup.nvertex if self.tup.nvertex is not None else '?'
        two = self.tup.ntrack if self.tup.ntrack is not None else '?'
        three = self.tup.necal if self.tup.necal is not None else '?'
        return f'({one}, {two}, {three})'


def calc_max_rel(vertex: pd.DataFrame, track: pd.DataFrame, ecal: pd.DataFrame) -> float:
    v, t, e = 0, 0, 0
    if 0 < len(vertex):
        v = np.max(
            np.abs([vertex.dx / vertex.x, vertex.dy / vertex.y, vertex.dz / vertex.z, vertex.dphi / vertex.phi]))
    if 0 < len(track):
        t = np.max(np.abs([track.dk / track.k, track.dtan_theta / track.tan_theta]))
    if 0 < len(ecal):
        e = np.max(np.abs([ecal.dx / ecal.x, ecal.dy / ecal.y, ecal.dz / ecal.z]))
    return np.max([v, t, e])
