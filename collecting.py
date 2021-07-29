"""
Collecting Data
"""
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
import pandas as pd
import collipy as cp
import matplotlib.pyplot as plt


def collecting_particle(alpha, particle, momentum, n, mode, threshold):
    filename = f'{particle}.pickle'
    path = os.path.join(os.path.dirname(os.path.normpath(__file__)), 'data', filename)
    if filename not in os.listdir(os.path.basename(path)):
        username = input('username: \n')
        password = input('password: \n')
        ac = cp.Accelerator(username, password, alpha)
        times = math.ceil(n/5)
        run = lambda m: ac.collect(particle, momentum, times, mode, threshold)
        with ThreadPoolExecutor() as executor:
            results = [executor.submit(run) for _ in range(5)]
            concurrent.futures.as_completed(results)
            # TODO: dismantle results to injections and cnts
        data = cp.InjectionCollection(alpha, particle, momentum, [inj for lst in results for inj in lst], mode, threshold, )
        data.dump(path)
        return data
    else:
        return cp.InjectionCollection.load(path)

