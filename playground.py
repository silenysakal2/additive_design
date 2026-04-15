# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy
from scipy.stats import qmc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from matplotlib import gridspec
from itertools import cycle

import ctypes
import math
import random
import platform
from tqdm.notebook import trange

from plotting import plot_2D_view


# %%
from design_criteria import wd2, cl2, Mm, phip, maxPro, latinize, evaluate

# %%
# %matplotlib widget

# %%
lib_name = "./maxpro.dll" if platform.system() == "Windows" else "./maxpro.so"
cppfn = ctypes.CDLL(lib_name)



cppfn.maxpro_design_meshgrid.argtypes = (
    ctypes.c_int, # nv
    ctypes.c_int, # ns
    ctypes.c_int, # seed
    ctypes.c_bool, # periodic
    ctypes.c_bool, # rand_ini
    ctypes.c_bool # rand_sel
)
cppfn.maxpro_design_meshgrid.restype = ctypes.POINTER(ctypes.c_int)

def maxpro_design_meshgrid(nv, ns, seed = None, periodic = True, rand_ini = True, rand_sel = True):
    global cppfn
    pointer = cppfn.maxpro_design_meshgrid(nv, ns, random.randrange(2 ** 31) if seed == None else seed, periodic, rand_ini, rand_sel)
    return np.ctypeslib.as_array(pointer, (ns, nv))


cppfn.gen_design_candidates.argtypes = (
    ctypes.c_char, # crit
    ctypes.c_int, # nv
    ctypes.c_int, # ns
    ctypes.c_longlong, # candidate_count
    ctypes.POINTER(ctypes.c_double), # candidates
    ctypes.c_int, # seed
    ctypes.c_bool, # periodic
    ctypes.c_bool # rand_sel
)

def gen_design_candidates(crit, nv, ns, seed = None, candidate_count = None, candidate_method = "monte-carlo", periodic = True, rand_sel = True):
    global cppfn
    if candidate_count == None: # Auto
        #candidate_count = min(16 ** nv * ns, 1024 ** 3 // nv) # 8 GB RAM limit for candidates
        candidate_count = min(10000 * ns, 1024 ** 3 // nv) # 8 GB RAM limit for candidates
    match candidate_method:
        case "monte-carlo":
            candidates = np.ascontiguousarray(np.random.rand(candidate_count, nv))
        case "meshgriddy":
            candidate_count = ns ** nv
            candidates = np.empty([ns] * nv + [nv])
            for v in range(nv):
                candidates[tuple(([slice(None)] * nv) + [v])] = ((np.arange(ns) + 0.5) / ns)[tuple(([np.newaxis] * v) + [slice(None)] + ([np.newaxis] * (nv - v - 1)))]
            candidates.shape = [ns ** nv, nv]
            candidates = np.ascontiguousarray(candidates)
        case default:
            raise ValueError("Unknown candidate method: \"" + candidate_method + "\"")
    match crit:
        case "maxpro":
            cppfn.gen_design_candidates(b'm', nv, ns, candidate_count, candidates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), random.randrange(2 ** 31) if seed == None else seed, periodic, rand_sel)
        case "phim":
            cppfn.gen_design_candidates(b'p', nv, ns, candidate_count, candidates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), random.randrange(2 ** 31) if seed == None else seed, periodic, rand_sel)
        case "maximin":
            cppfn.gen_design_candidates(b'M', nv, ns, candidate_count, candidates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), random.randrange(2 ** 31) if seed == None else seed, periodic, rand_sel)
    return candidates[:ns, :]


cppfn.maxpro_addPoint_semiAnalytical.argtypes = (
    ctypes.c_int, # nv
    ctypes.c_int, # ns
    ctypes.POINTER(ctypes.c_double), # points
    ctypes.c_double, # error_treshold
    ctypes.c_int, # min_iterations
    ctypes.c_int, # max_iterations
    ctypes.c_bool # periodic
)
cppfn.maxpro_addPoint_semiAnalytical.restype = ctypes.c_longlong



def maxpro_addPoint_semiAnalytical(points: np.ndarray, min_iterations = 2, max_iterations = 16, error_treshold = 1e-6, periodic = True) -> np.ndarray:
    global cppfn
    assert my_des.flags['C_CONTIGUOUS']
    assert my_des.dtype == np.float64
    ns, nv = points.shape
    points2 = np.empty((ns + 1, nv), dtype=np.float64) # There isn't much of a better way to do this than to copy the array
    assert points2.flags['C_CONTIGUOUS']
    points2[:ns] = points
    skipped = cppfn.maxpro_addPoint_semiAnalytical(nv, ns, points2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), error_treshold, min_iterations, max_iterations, periodic)
    #print("Ns = " + str(ns) +", Skipped " + str(skipped) + "/" + str(ns ** nv) + " (" + str(round(skipped / (ns ** nv) * 100)) + " %)")
    return points2

'''
# version with OpenMP:

cppfn.maxpro_addPoint_semiAnalytical_Par.argtypes = (
    ctypes.c_int, # nv
    ctypes.c_int, # ns
    ctypes.POINTER(ctypes.c_double), # points
    ctypes.c_double, # error_treshold
    ctypes.c_int, # min_iterations
    ctypes.c_int, # max_iterations
    ctypes.c_bool # periodic
)
cppfn.maxpro_addPoint_semiAnalytical_Par.restype = ctypes.c_longlong

def maxpro_addPoint_semiAnalytical_Par(points: np.ndarray, min_iterations = 2, max_iterations = 16, error_treshold = 1e-6, periodic = True) -> np.ndarray:
    global cppfn
    assert my_des.flags['C_CONTIGUOUS']
    assert my_des.dtype == np.float64
    ns, nv = points.shape
    points2 = np.empty((ns + 1, nv), dtype=np.float64) # There isn't much of a better way to do this than to copy the array
    assert points2.flags['C_CONTIGUOUS']
    points2[:ns] = points
    skipped = cppfn.maxpro_addPoint_semiAnalytical_Par(nv, ns, points2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), error_treshold, min_iterations, max_iterations, periodic)
    print("Ns = " + str(ns) +", Skipped " + str(skipped) + "/" + str(ns ** nv) + " (" + str(round(skipped / (ns ** nv) * 100)) + " %)")
    return points2

'''


# %%
nv = 2
ns = 100


# Additive designs
periodic = True
rand_ini = True
rand_sel = False

# qmc
scramble = False 

# %%
my_des = np.random.rand(5, nv)

# %%
my_des = maxpro_addPoint_semiAnalytical(my_des, 1, 1000, 1e-6, True)


# %%
my_des

# %%
my_des.shape

# %%
plt.close("all")

while my_des.shape[0] < ns:
    my_des = maxpro_addPoint_semiAnalytical(my_des, 1, 100, 1e-6, True)


if nv==2:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_2D_view(my_des.shape[1], my_des.shape[0], my_des, ax, vars_to_plot=[0, 1])
else:

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax0 = ax[0][0]

    ax0.scatter(my_des[:, 0], my_des[:, 1], c = "k")
    ax0.scatter(my_des[-1, 0], my_des[-1, 1], c = "red")
    ax0.set_xlim(0, 1)
    ax0.set_xticklabels(["" for _ in range(my_des.shape[0])])
    ax0.set_ylim(0, 1)
    ax0.set_yticklabels(["" for _ in range(my_des.shape[0])])
    ax0.set_xticks(my_des[:, 0])
    ax0.set_yticks(my_des[:, 1])

    ax1 = ax[1][0]
    ax1.scatter(my_des[:, 1], my_des[:, 2], c = "k")
    ax1.scatter(my_des[-1, 1], my_des[-1, 2], c = "red")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(my_des[:, 1])
    ax1.set_yticks(my_des[:, 2])
    ax2 = ax[0][1]
    ax2.scatter(my_des[:, 2], my_des[:, 0], c = "k")
    ax2.scatter(my_des[-1, 2], my_des[-1, 0], c = "red")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(my_des[:, 2])
    ax2.set_yticks(my_des[:, 0])

    fig.show()



# %% [markdown]
# # Batch-generating designs

# %%
# Taken from uMaxPro on GitHub, slightly modified

def maxPro_np(x: np.ndarray, periodic = True) -> float:  # single loop
    """
    Compute the (u)MaxPro criterion for a given design matrix.

    This function calculates the MaxPro or uMaxPro criterion for a given 2D design matrix `x`.
    The MaxPro criterion is used in experimental design to ensure good space-filling properties.
    If `periodic` is set to True, the function computes the uMaxPro criterion, which accounts
    for periodic boundary conditions.

    Args:
        x (np.ndarray): A 2D array of shape (ns, nv) representing the design points.
        ns (int): The number of samples (design points).
        nv (int): The number of variables (dimensions).
        periodic (bool, optional): If True, computes the uMaxPro criterion
            (periodic case). If False, computes the MaxPro criterion
            (non-periodic case). Default is False.

    Returns:
        float: The computed (u)MaxPro criterion value.

    Notes:
        - The MaxPro criterion favors designs that are space-filling by maximizing the
          minimum product of squared distances between points.
        - The uMaxPro variant uses periodic distance calculations.
    """

    # Ensure ns and nv are consistent with the shape of x
    ns, nv = x.shape

    maxpro = 0  # Initialize the criterion accumulator

    # Iterate over each design point
    for i in range(ns):
        # Compute the absolute differences between point i and all previous points
        deltas = np.abs(x[i, :] - x[0:i, :])

        if periodic is True:
            # Apply periodic boundary conditions by wrapping distances
            deltas = np.minimum(deltas, 1 - deltas)

        # Square the differences to get squared distances
        dsq = deltas ** 2

        # Compute the reciprocal of the product of squared distances for each pair
        # Sum them up and add to the maxpro accumulator
        maxpro += np.sum(1. / np.prod(dsq, axis=1))

    return maxpro


# %%
class Setting:
    nv: int
    ns: int
    nr: int
    coord: list[int]

    def __init__(self, nv, ns, nr, coord):
        self.nv = nv
        self.ns = ns
        self.nr = nr
        self.coord = coord

    def __str__(self):
        return f"nv: {self.nv} ns: {self.ns} nr: {self.nr} coord: {self.coord}"


# %%
settings = []
nr = 10
ns = 1024
for nvar in range(2, 3):
    for i, nsim in enumerate([ns]):
        settings.append(Setting(nvar, nsim, nr, [nvar-2, i]))
        print(settings[-1])

# %% [markdown]
# # Meshgrid design here

# %%
design = (maxpro_design_meshgrid(nv, ns, seed = None, periodic = True, rand_ini = True, rand_sel = True) + 0.5) / ns

# %%
fig, ax = plt.subplots(1, 1, figsize = [16, 16])
ax.scatter(design[:, 0], design[:, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.show()

# %%
import numpy as np
import threading
from queue import Queue
from tqdm.auto import tqdm

def make_initial_design(ns, nv, rng):
    curr_des = np.empty((ns, nv))
    for v in range(nv):
        coords = rng.permutation((np.arange(ns) / ns) + 0.5/ns)
        curr_des[:, v] = coords
    return curr_des


def worker(worker_id, task_queue, results, nv, ns, ns_ini, min_iters, max_iters, error_treshold, periodic, pbar_lock):
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        i, seed = item
        rng = np.random.default_rng(seed)

        curr_des = make_initial_design(ns_ini, nv, rng)

        # one persistent bar position per worker
        bar = tqdm(
            total=ns,
            initial=curr_des.shape[0],
            desc=f"thread {worker_id} | design {i}",
            position=worker_id,
            leave=True,
            dynamic_ncols=True,
        )

        while curr_des.shape[0] < ns:
            old_n = curr_des.shape[0]
            curr_des = maxpro_addPoint_semiAnalytical(curr_des, min_iters, max_iters, error_treshold, periodic)
            new_n = curr_des.shape[0]

            # usually +1, but safe even if function adds more than one row
            if new_n > old_n:
                with pbar_lock:
                    bar.update(new_n - old_n)

        maxpro_val = maxPro_np(curr_des)
        results[i] = (curr_des, maxpro_val)

        with pbar_lock:
            bar.set_description(f"thread {worker_id} | done {i}")
            bar.close()

        task_queue.task_done()


# choose how many bars / worker threads you want
n_threads = 10



#maxpros = -np.ones([4, 9, nr])  # only if nr is same for all settings


for setting in settings:
    print(setting)
    nv = setting.nv
    ns = setting.ns
    nr = setting.nr
    ns_ini = 5
    min_iters, max_iters = 1,1
    error_treshold = 1e-8
    periodic = True

    designs = np.empty((nr, ns, nv))
    maxpro_vals = np.empty(nr)

    task_queue = Queue()
    results = [None] * nr
    pbar_lock = threading.Lock()

    # fill task queue
    for i in range(nr):
        task_queue.put((i, 12345 + i))

    # add stop signals
    for _ in range(n_threads):
        task_queue.put(None)

    # start workers
    threads = []
    for worker_id in range(n_threads):
        t = threading.Thread(
            target=worker,
            args=(worker_id, task_queue, results, nv, ns, ns_ini, min_iters, max_iters, error_treshold, periodic, pbar_lock),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # wait until all work is done
    task_queue.join()

    for t in threads:
        t.join()

    # collect results
    for i, item in enumerate(results):
        curr_des, maxpro_val = item
        designs[i] = curr_des
        maxpro_vals[i] = maxpro_val
        # maxpros[tuple(setting.coord + [i])] = maxpro_val

    np.save(f"data/designs_nv{nv:02d}_ns{ns:04d}_nr{nr:04d}.npy", designs)

# %% [raw]
# maxpros = -np.ones([4, 9, nr]) # So that there are negative ones for uninitialized
#
# for setting in settings:
#     print(setting)
#     nv = setting.nv
#     ns = setting.ns
#     nr = setting.nr
#
#     designs = np.empty([nr, ns, nv])
#     for i in trange(nr):
#         curr_des = np.empty([5, nv])
#         for v in range(nv):
#             coords = np.random.permutation((np.arange(5) / 5) + (0.1))
#             curr_des[:, v] = coords
#         #print(curr_des)
#         while curr_des.shape[0] < ns:
#             curr_des = maxpro_addPoint_semiAnalytical(curr_des, 1, 100, 1e-8, True)
#         designs[i] = curr_des
#         maxpros[tuple(setting.coord + [i])] = maxPro_np(curr_des)
#     np.save(f"data/designs_nv{"{:02d}".format(nv)}_ns{"{:04d}".format(ns)}_nr{"{:04d}".format(nr)}.npy", designs)

# %%

# %%
from pathlib import Path
import re
import numpy as np

def load_designs_SA(nv, folder="data_SA"):
    folder = Path(folder)

    # nv=0002_ns=00512_per=True_ndes=000401-x_opt_all.npy
    pattern = re.compile(
        rf"nv={nv:04d}_ns=(\d{{5}})_per=True_ndes=000401-x_opt_all\.npy"
    )

    files_with_ns = []

    for path in folder.glob(f"nv={nv:04d}_ns=*_per=True_ndes=000401-x_opt_all.npy"):
        m = pattern.fullmatch(path.name)
        if m:
            ns = int(m.group(1))
            files_with_ns.append((ns, path))

    files_with_ns.sort(key=lambda x: x[0])

    ns_values = [ns for ns, _ in files_with_ns]
    designs_SA = [np.load(path) for _, path in files_with_ns]

    return ns_values, designs_SA


# %%
# LOAD ADDITIVE DESIGNS

# nr =
# nv =

designs = np.load(f"data/designs_nv{nv:02d}_ns{ns:04d}_nr{nr:04d}.npy")


# LOAD SA DESIGNS

ns_values, designs_SA = load_designs_SA(nv)

print(ns_values)
print(len(designs_SA))
print(designs_SA[0].shape if designs_SA else "nic nenalezeno")

# %%
designs.shape, designs_SA

# %%
maxpros_SA_x = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])

# %%
maxpros_SA = np.empty([401, maxpros_SA_x.shape[0]])

# %%
for ks in trange(maxpros_SA_x.shape[0]):
    for r in range(401):
        maxpros_SA[r, ks] = maxPro_np(designs_SA[ks][r])

# %%
# Parallel version with selection of some sample sizes only

import numpy as np
import threading
from queue import Queue
from tqdm.auto import tqdm

def worker(worker_id, task_queue, designs, sample_sizes, maxpros, maxprosL, pbar_lock):
    bar = tqdm(
        total=len(sample_sizes),
        initial=0,
        desc=f"thread {worker_id}",
        position=worker_id,
        leave=True,
        dynamic_ncols=True,
    )

    while True:
        r = task_queue.get()
        if r is None:
            task_queue.task_done()
            break

        with pbar_lock:
            bar.reset(total=len(sample_sizes))
            bar.n = 0
            bar.set_description(f"thread {worker_id} | r={r}")
            bar.refresh()

        for j, ns in enumerate(sample_sizes):
            d = designs[r, :ns, :]
            maxpros[r, j] = maxPro_np(d)
            maxprosL[r, j] = maxPro_np(latinize(d))

            with pbar_lock:
                bar.update(1)

        task_queue.task_done()

    with pbar_lock:
        bar.close()


n_threads = designs.shape[0]
n_discr = 30

sample_sizes = np.unique(np.linspace(5, designs.shape[1], n_discr, dtype=int))

maxpros = -np.ones((designs.shape[0], len(sample_sizes)))
maxprosL = -np.ones((designs.shape[0], len(sample_sizes)))

task_queue = Queue()
pbar_lock = threading.Lock()

for r in range(designs.shape[0]):
    task_queue.put(r)

for _ in range(n_threads):
    task_queue.put(None)

threads = []
for worker_id in range(n_threads):
    t = threading.Thread(
        target=worker,
        args=(worker_id, task_queue, designs, sample_sizes, maxpros, maxprosL, pbar_lock),
        daemon=True,
    )
    t.start()
    threads.append(t)

task_queue.join()

for t in threads:
    t.join()





fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for r in range(designs.shape[0]):
    ax.plot(sample_sizes, maxpros[r], c='r', label='additive' if r == 0 else None)
    ax.plot(sample_sizes, maxprosL[r], c='k', label='additive latinized' if r == 0 else None)

plt.legend()
plt.show()

# %%
maxpros.shape, maxprosL.shape

# %% [raw]
# maxpros = -np.ones([designs.shape[0], designs.shape[1]+1])
# maxprosL = -np.ones([designs.shape[0], designs.shape[1]+1])
#
# for r in trange(designs.shape[0]):
#     for ns in range(5, designs.shape[1]+1):
#         d = designs[r, :ns, :]
#         maxpros[r, ns] = maxPro_np(d)
#         maxprosL[r, ns] = maxPro_np(latinize(d))

# %%
ns = maxpros.shape[1]
ns

# %%
exponent = 3.15 #2D
#exponent = 3.3 #3D

maxpros2 = maxpros / (sample_sizes ** exponent)
maxpros2L = maxprosL / (sample_sizes ** exponent)
maxpros_SA2 = maxpros_SA / (maxpros_SA_x ** exponent)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax0 = ax

for r in range(designs.shape[0]):
    ax0.plot(sample_sizes, maxpros2[r] , c='r', label='additive' if r == 0 else None )
    ax0.plot(sample_sizes, maxpros2L[r], c='k', label='additive latinized' if r == 0 else None)

for r in range(401):
    ax0.plot(maxpros_SA_x, maxpros_SA2[r], c="b",
             label='SA' if r == 0 else None)

plt.legend()
plt.show()

# %%
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax0 = ax[0][0]

my_des = designs[5]
ns, nv = my_des.shape

if nv==2:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_2D_view(my_des.shape[1], my_des.shape[0], my_des, ax, vars_to_plot=[0, 1])
else:
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax0 = ax[0][0]

    ax0.scatter(my_des[:, 0], my_des[:, 1], c = "k")
    ax0.scatter(my_des[-1, 0], my_des[-1, 1], c = "red")
    ax0.set_xlim(0, 1)
    ax0.set_xticklabels(["" for _ in range(my_des.shape[0])])
    ax0.set_ylim(0, 1)
    ax0.set_yticklabels(["" for _ in range(my_des.shape[0])])
    ax0.set_xticks(my_des[:, 0])
    ax0.set_yticks(my_des[:, 1])

    ax1 = ax[1][0]
    ax1.scatter(my_des[:, 1], my_des[:, 2], c = "k")
    ax1.scatter(my_des[-1, 1], my_des[-1, 2], c = "red")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(my_des[:, 1])
    ax1.set_yticks(my_des[:, 2])
    ax2 = ax[0][1]
    ax2.scatter(my_des[:, 2], my_des[:, 0], c = "k")
    ax2.scatter(my_des[-1, 2], my_des[-1, 0], c = "red")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(my_des[:, 2])
    ax2.set_yticks(my_des[:, 0])

    fig.show()

# %%
print(f" Nsim = {len(my_des)}" )

# %%
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(my_des[:, 0], my_des[:, 1], my_des[:, 2])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)


# %%
class Design:
    name: str
    points: np.ndarray
    maxpro: float
    umaxpro: float
    maximin_nper: float
    maximin: float
    discr: float
    phip: float
    pphip: float
    wd2: float
    cl2: float

    def comp_stats(self):
        self.maxpro = maxPro(self.points, False)
        self.umaxpro = maxPro(self.points, True)
        self.maximin = Mm(self.points, False)
        self.pmaximin = Mm(self.points, True)
        self.phip = phip(self.points, False)
        self.pphip = phip(self.points, True)
        self.wd2 = wd2(self.points)
        self.cl2 = cl2(self.points)
        self.mdiscr = scipy.stats.qmc.discrepancy(self.points, method='MD' ) 
        self.l2stardiscr = scipy.stats.qmc.discrepancy(self.points, method='L2-star' ) 
        #CD: Centered Discrepancy - subspace involves a corner of the hypercube
        #WD: Wrap-around Discrepancy - subspace can wrap around bounds
        #MD: Mixture Discrepancy - mix between CD/WD covering more criteria
        #L2-star: L2-star discrepancy - like CD BUT variant to rotatio

    def __init__(self, points, name):
        self.name = name
        self.points = points
        self.comp_stats()

METHODS = [
    "as_maxpro", # Adaptive sampling; already latinized
    "as_phim",
    "as_maximin",
    "mg_maxpro", # Meshgrid
    "qmc_sobol", # Quasi Monte Carlo; already latinized
    "qmc_halton"
]

designs = dict()

# %%
# %%time
designs["mg_maxpro"] = Design(
    (maxpro_design_meshgrid(nv, ns, periodic = periodic, rand_ini = rand_ini, rand_sel = rand_sel).astype(float) + 0.5) / ns,
    "mg_maxpro"
)

# %%
# %%time
designs["as_maxpro"] = Design(
    latinize(gen_design_candidates("maxpro", nv, ns, candidate_method = "monte-carlo", periodic = periodic, rand_sel = rand_sel)),
    "as_maxpro",
)

# %%
# %%time
designs["as_phim"] = Design(
    latinize(gen_design_candidates("phim", nv, ns, candidate_method = "monte-carlo", periodic = periodic, rand_sel = rand_sel)),
    "as_maxpro",
)

# %%
# %%time
designs["as_maximin"] = Design(
    (gen_design_candidates("maximin", nv, ns, candidate_method = "monte-carlo", periodic = periodic, rand_sel = rand_sel)),
    "as_maxpro",
)

# %%
# %%time
sobol_sampler = qmc.Sobol(d = nv, scramble = scramble)
sobol_points = sobol_sampler.random(ns)
sobol_points -= sobol_points.min()
sobol_points += 0.5 * (1 - sobol_points.max())
designs["qmc_sobol"] = Design(latinize(sobol_points), "qmc_sobol")

# %%
# %%time
halton_sampler = qmc.Halton(d = nv, scramble = scramble)
halton_points = halton_sampler.random(ns)
halton_points -= halton_points.min()
halton_points += 0.5 * (1 - halton_points.max())
designs["qmc_halton"] = Design(latinize(halton_points), "qmc_halton")

# %%
best_maxpro = math.inf
best_umaxpro = math.inf
best_maximin = 0
best_pmaximin = 0
best_mdiscr = math.inf
best_phip = math.inf
best_pphip = math.inf
best_cl2 = math.inf
best_wd2 = math.inf

for method in METHODS:
    best_maxpro = min(best_maxpro, designs[method].maxpro)
    best_umaxpro = min(best_umaxpro, designs[method].umaxpro)
    best_maximin = max(best_maximin, designs[method].maximin)
    best_pmaximin = max(best_pmaximin, designs[method].pmaximin)
    best_mdiscr = min(best_mdiscr, designs[method].mdiscr)
    best_phip = min(best_phip, designs[method].phip)
    best_pphip = min(best_pphip, designs[method].pphip)
    best_cl2 = min(best_cl2, designs[method].cl2)
    best_wd2 = min(best_wd2, designs[method].wd2)

CELL_WIDTH = 25
print((("+" + ("-" * (CELL_WIDTH-1))) * 10) + "+")
print("| name\t| MaxPro\t| uMaxPro\t| maximin\t| pmaximin\t| mixed discrepancy\t| phip\t| pphip\t| cl2\t| wd2\t|".expandtabs(CELL_WIDTH))
print("=" * CELL_WIDTH * 10 + "=")
for i, method in enumerate(METHODS):
    print((
        "| " +
        method +
        "\t| " +
        str(designs[method].maxpro / best_maxpro) +
        "\t| " +
        str(designs[method].umaxpro / best_umaxpro) +
        "\t| " +
        str(best_maximin / designs[method].maximin) +
        "\t| " +
        str(best_pmaximin / designs[method].pmaximin) +
        "\t| " +
        str(designs[method].mdiscr / best_mdiscr) +
        "\t| " +
        str(designs[method].phip / best_phip) +
        "\t| " +
        str(designs[method].pphip / best_pphip) +
        "\t| " +
        str(designs[method].cl2 / best_cl2) +
        "\t| " +
        str(designs[method].wd2 / best_wd2) +
        "\t|"
    ).expandtabs(CELL_WIDTH))
    print((("+" + ("-" * (CELL_WIDTH-1))) * 10) + "+")

print("\n\n\n\n")
    
print((("+" + ("-" * (CELL_WIDTH-1))) * 10) + "+")
print("| name\t| MaxPro\t| uMaxPro\t| maximin\t| pmaximin\t| mixed discrepancy\t| phip\t| pphip\t| cl2\t| wd2\t|".expandtabs(CELL_WIDTH))
print("=" * CELL_WIDTH * 10 + "=")
for i, method in enumerate(METHODS):
    print((
        "| " +
        method +
        "\t| " +
        str(designs[method].maxpro) +
        "\t| " +
        str(designs[method].umaxpro) +
        "\t| " +
        str(designs[method].maximin) +
        "\t| " +
        str(designs[method].pmaximin) +
        "\t| " +
        str(designs[method].mdiscr) +
        "\t| " +
        str(designs[method].phip) +
        "\t| " +
        str(designs[method].pphip) +
        "\t| " +
        str(designs[method].cl2) +
        "\t| " +
        str(designs[method].wd2) +
        "\t|"
    ).expandtabs(CELL_WIDTH))
    print((("+" + ("-" * (CELL_WIDTH-1))) * 10) + "+")


# %%
def plot_des(ax, des, name, color='k', alpha=1):
    ax.scatter(des[:, -2], des[:, -1], color=color, alpha=alpha)
    ax.scatter(des[0, -2], des[0, -1], color="red", alpha=alpha)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(name, fontsize=10, pad=2)
    
    tick_length = 0.005
    for x in des[:, 0]:
        ax.plot([x, x], [0, tick_length], color='black', linewidth=0.5, alpha=alpha)
    for y in des[:, 1]:
        ax.plot([0, tick_length], [y, y], color='black', linewidth=0.5, alpha=alpha)

# Calculate layout
num_designs = len(designs)
cols = 2
rows = math.ceil(num_designs / cols)

fig, axs = plt.subplots(
    rows, cols, figsize=(4 * cols, 4 * rows),
    gridspec_kw=dict(wspace=0.05, hspace=0.15)  # Tight spacing
)
axs = axs.flatten()


for pts in range (ns,ns+1):
    for i, (name, design) in enumerate(designs.items()):
        plot_des(axs[i], design.points[:pts,:], name)

    # Turn off any unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    plt.show()



# %%
nr = 1000
hist = np.zeros([ns] * nv, dtype=int)
for i in trange(nr):
    des = maxpro_design_meshgrid(nv, ns, seed=i, periodic = periodic, rand_ini = rand_ini, rand_sel = rand_sel)
    np.add.at(hist, (des[:, 0], des[:, 1]), 1)

# %%
hist

# %%

# %%
hist2 = hist.astype(float) / nr * ns
hist2

# %%
symmetrize = False # to generate the additional designs from the existing ones

n_des = nr #number of designes packed into x_opt_all


freq = hist

print(f"Check that the no of data in bins {np.sum(freq)} matches the number of points {n_des*ns}\n")
print(f"\nNow the symmetrized histogram construction")

design_count_multiplier = 1
freq_s = freq.copy() # Stands for frequencies symmetrized (identical to `freq` if not symmetrizing)
if symmetrize:
    # Account for reordering the axes
    for set_v in range(nv - 1): # Note that this trange is going to speed up over time
        freq_copy = np.copy(freq_s) # Add to the `freq_s` array; this is for a reference to get all the distinct transposed versions of the original
        transpose_indices = list(range(nv))
        for swapped_v in range(set_v + 1, nv): # You would normally go from `set_v` + 0, but I am adding to an array equivalent to `swapped_v` = `set_v`
            transpose_indices[set_v], transpose_indices[swapped_v] = swapped_v, set_v
            freq_s += np.transpose(freq_copy, tuple(transpose_indices))
            transpose_indices[swapped_v] = swapped_v # Cleanup for next loop run
    
    # Mirror about every axis
    for v in range(nv):
        freq_s += np.flip(freq_s, axis=v)


    design_count_multiplier = (2 ** nv) * math.factorial(nv)
    print("Design count incerased ", design_count_multiplier," times from ",n_des, " to ", n_des * design_count_multiplier)
    
    print(f"Check that the no of data in bins {np.sum(freq_s)} matches the number of points {n_des*ns*design_count_multiplier}")

# Recompute frequencies into relative freq (ave density)
histogram = freq * ((ns ** (nv-1)) / n_des)
histogram_s = freq_s * ((ns ** (nv-1)) / n_des / design_count_multiplier)

print(f"\n")
print(f"Minimum density: {histogram.min()},  maximum density: {histogram.max()}, stdev of density: {histogram.std()}")



     

# %%
fig, ax = plt.subplots(figsize=(16, 16))

vmin, vmax = 0, 5

if periodic:
    vmin, vmax = 0, 2
    
ax.matshow(histogram_s, vmin=vmin, vmax=vmax)

plt.show()

# %%
phimc[:2]

# %%
n = 0

# %%
#pts = points[:, (0, 1)]
pts = phimc[:n]
n += 1
# Assume pts is a (N, 2) NumPy array with values in [0, 1]
offsets = [-1, 0, 1]
tiled_points = []

for dx in offsets:
    for dy in offsets:
        shifted = pts + np.array([dx, dy])
        tiled_points.append(shifted)

tiled_points = np.vstack(tiled_points)

# Plot
fig, ax = plt.subplots(figsize=(16, 16))
ax.scatter(tiled_points[:, 0], tiled_points[:, 1], s=5)

# Set limits and aspect
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')

# Tile boundary lines
for x in range(-1, 3):
    ax.axvline(x, color='gray', linestyle='--', linewidth=0.8)
for y in range(-1, 3):
    ax.axhline(y, color='gray', linestyle='--', linewidth=0.8)

# Add projections:
tick_length = 0.02  # Length of the projection lines

# Vertical ticks at bottom (y = -1)
for x in tiled_points[:, 0]:
    ax.plot([x, x], [-1, -1 + tick_length], color='black', linewidth=0.5)

# Horizontal ticks at left (x = -1)
for y in tiled_points[:, 1]:
    ax.plot([-1, -1 + tick_length], [y, y], color='black', linewidth=0.5)

plt.show()


# %% [markdown]
# # Gradient descent of design
#
#
# ## Maxpro (`gradientDescent_maxpro`)
#
# Time complexity: $n_s ^2 \cdot n_v ^2 \cdot n_\mathrm{steps}$, space complexity: $n_s ^2 \cdot n_v ^2$

# %%
def gradientDescent_maxpro(input_des: np.ndarray, steps: np.ndarray = np.pow(1.1, -np.arange(64)) * 0.001, copy: bool = True) -> np.ndarray:
    design = input_des.copy() if copy else input_des
    ns = design.shape[0]
    nv = design.shape[1]

    for step_size in steps:
        deltas = design[:, np.newaxis, :] - design[np.newaxis, :, :]
        deltas[deltas > 0.5 ] -= 1 # Periodize
        deltas[deltas < -0.5] += 1
        d_sq = deltas * deltas
        d_cube = (d_sq * deltas)
        deltas_aranged = d_sq[:, :, :, np.newaxis] + np.zeros([nv])[np.newaxis, np.newaxis, np.newaxis, :] # Add the zeros to reshape (copy the values along the last dimension) (yes, there is no better way to do that)
        deltas_aranged[:, :, np.arange(nv), np.arange(nv)] = d_cube
        deltas_aranged[np.arange(ns), np.arange(ns), :, :] = np.inf # It will be divided by, they should be zeroed
        derivatives = np.sum(1 / np.prod(deltas_aranged, axis = 3), axis = 1) # NOTE: They are halved and negated (for efficiency)
        derivatives *= 1 / np.max(np.abs(derivatives)) # They kept overflowing the float64 max, that's why they're getting divided twice
        max_derivative = np.sqrt(np.max(np.sum(derivatives * derivatives, axis = 1)))
        design += derivatives * (step_size / max_derivative)
        design[design < 0] += 1 # Periodize
        design[design > 1] -= 1

    return design


# %%
np.sum(np.pow(1.005, -np.arange(256)) * 0.001), np.pow(1.005, -np.arange(256)) * 0.001

# %%
my_design = designs["as_maxpro"].points

# %%
step_count = 64
step_min_relative = 1e-6 # This is basically the precision
steps = np.pow(step_min_relative ** (1 / step_count), np.arange(step_count)) * (0.25 / ns)
print("Next step factor (should be over 0.5):", (step_min_relative ** (1 / step_count)))

# %%
steps

# %%
ns

# %%
(1/106) ** 2

# %%
my_design2 = gradientDescent_maxpro(my_design, steps, True)

# %%
#my_design2 = my_design

# %%
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(my_design[:, 0],  my_design[:, 1],  c = "r", alpha = 0.5)
ax.scatter(my_design2[:, 0], my_design2[:, 1], c = "g")

min_diff = 1
for v in range(nv):
    sorted = np.sort(my_design2[:, v])
    diff = np.min(np.abs(sorted - np.roll(sorted, 1))) # It's the local minimum, even though min is not in the name, not to be confused
    if min_diff > diff:
        min_diff = diff
#print("Min diff:", min_diff)

# %% [markdown]
# # Playing around with golden ratio

# %%
ns = 1

# %%
#design = np.mod(np.arange(ns) * ((1 + np.sqrt(5)) / 2), 1)
design = np.mod(np.log2(np.arange(ns) + 0.5), 1)

fig, ax = plt.subplots(figsize=(16, 1))

ax.scatter(design,   np.zeros_like(design))
ax.scatter(design+1, np.zeros_like(design))
ax.scatter(design-1, np.zeros_like(design))
ax.set_xlim([-1, 2])

ns += 1

# %%
plt.close("all")

# %% [markdown]
# # Normalizing MaxPro
#
# It generally seems to be about simply dividing by $n_s^3$, but I don't quite have a nice derivation of this in higher dimensions.
#
# ## 1D definitive derivation
#
# Consider a *good* design with $n_s$ points. View the design from the perspective of a single point, $X$, and its contribution to MaxPro. That would be:
#
# $$\sum _{i \neq X} ^{n_s} \frac{1}{\delta _{i, X} ^2}$$
#
# Now, increase the number of samples by $k > 1$; we want something like $\lim \limits _{k \rightarrow 1}$ (in other words $k$ should be really small). For the new design to be similarly good, we'd expect the points around $X$ to go $k$ times closer (from either side). For these points, the $\frac{1}{\delta ^2}$ may be of large values because of division by zero: even for small $\Delta x$, $\int _{-x} ^{x} \frac{1}{x^2}$ is *infinite* (formally undefined). For that reason, in the case of these nearby points, it's necessary to track their exact positions rather than a general density.
#
# For these shrinked points, their MaxPro contribution with $X$ will increase $k^2$ times.
#
# As for the new points: for $k$ close enough to $1$, we can say all these points are $0.5$ away from $X$. There are $n_s \left(k - 1\right)$ new points. That brings the MaxPro contribution to:
#
# $$\sum ^{n_s \left(k - 1\right)} \frac{1}{0.5 ^2} = 4 n_s \left(k - 1\right)$$
#
# Finally, all of this has to be $k$ times because there are now $k$ times more points that contribute like $X$ to the total MaxPro.
#
# If we add $\mathrm{d} n_s$ points, $k = 1 + \frac{\mathrm{d} n_s}{n_s}$. Plug that in, we get a differential equation (let $M\left(n_s\right)$ be MaxPro of a design):
#
# $$
# \begin{aligned}
# \mathrm{d} M\left(n_s\right) &=
# k \cdot \left(M\left(n_s\right) \cdot k^2 + 4 n_s \mathrm{d} n_s\right) - M\left(n_s\right) \\
# &=
# \left(1 + \frac{\mathrm{d} n_s}{n_s}\right) \cdot \left(M\left(n_s\right) \cdot \left(1 + 2 \frac{\mathrm{d} n_s}{n_s} + \frac{\mathrm{d} n_s ^2}{n_s ^2}\right) + 4 n_s \mathrm{d} n_s\right) - M\left(n_s\right) \\
# &=
# \left(1 + \frac{\mathrm{d} n_s}{n_s}\right) \cdot \left(M\left(n_s\right) + \frac{2 M\left(n_s\right) \mathrm{d} n_s}{n_s} + 4 n_s \mathrm{d} n_s\right) - M\left(n_s\right) \\
# &=
# M\left(n_s\right) + \frac{2 M\left(n_s\right) \mathrm{d} n_s}{n_s} + 4 n_s \mathrm{d} n_s + \frac{M\left(n_s\right) \mathrm{d} n_s}{n_s} + \frac{2 M\left(n_s\right) \mathrm{d} n_s ^2}{n_s ^2} + 4 \mathrm{d} n_s ^2 - M\left(n_s\right) \\
# &=
# \frac{3 M\left(n_s\right) \mathrm{d} n_s}{n_s} + 4 n_s \mathrm{d} n_s \\
# \frac{\mathrm{d} M\left(n_s\right)}{\mathrm{d} n_s} &=
# \frac{3 M\left(n_s\right)}{n_s} + 4 n_s
# \end{aligned}
# $$
#
# Wolfram Alpha says that's:
#
# $$M \left(n_s\right) = c n_s ^3 - 4 n_s ^2$$
#
# Where $c$ appears to be, from testing, $\frac{\pi ^2}{3}$.

# %%
ns = 1

print("ns\tLHS 1D MaxPro\tnormalized\tnormalized but only ns^3, not that linear component".expandtabs(24))
while ns < 1_000_000_000:
    actual_maxpro = ns * (2 * np.sum(1 / (((np.arange((ns-1) // 2) + 1) / ns) ** 2)) + (0 if (ns % 2 == 1) else 4))
    normalized1 = actual_maxpro / ((ns ** 3) * (np.pi ** 2) / 3 - (4 * (ns ** 2)))
    normalized2 = actual_maxpro / (ns ** 3) / (np.pi ** 2) * 3
    print(f"{ns}\t{actual_maxpro}\t{normalized1}\t{normalized2}".expandtabs(24))
    ns *= 2

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
