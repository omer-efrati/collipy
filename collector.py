"""
Collecting Data
"""
import os
import math
import concurrent.futures
from collections import Counter
import numpy as np
import pandas as pd
import collipy as cp
import matplotlib.pyplot as plt


def collecting_particle(alpha, particle, momentum, n, mode, threshold):
    filename = f'{particle}.pickle'
    path = os.path.join(os.path.dirname(os.path.normpath(__file__)), 'data', filename)
    x = os.path.dirname(path)
    if filename not in os.listdir(os.path.dirname(path)):
        # username = input('username: \n')
        # password = input('password: \n')
        username = 'omerefrati'
        password = 'zdHuu%#4@S^2'
        ac = cp.Accelerator(username, password, alpha)
        trds = 8
        times = math.ceil(n/trds)
        run = lambda: ac.collect(particle, momentum, times, mode, threshold)
        injections = []
        cnts = Counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(run) for _ in range(trds)]
            for f in concurrent.futures.as_completed(results):
                injs, cnt = f.result()
                injections += injs
                cnts += cnt
        data = cp.InjectionCollection(alpha, particle, momentum, injections, mode, threshold, cnts)
    return data
    #     data.dump(path)
    #     return data
    # else:
    #     return cp.InjectionCollection.load(path)
