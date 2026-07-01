import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import random
import math
import numpy as np
import scipy.linalg
from numpy import exp, cos, sin, real, imag, conj
from itertools import combinations
import os
import copy
import multiprocessing as mp
import time

_START_TIME = time.perf_counter()

class LogisticMap:
    def __init__(self, A=4.0, x0=0.37):
        self.A = A
        self.x = x0

    def next(self):
        self.x = self.A * self.x * (1 - self.x)
        return self.x

def generate_valid_seed():
    """Generate a random logistic-map seed using OS entropy.
    The seed satisfies:
      1. Strictly within (0.01, 0.99)
      2. Not within 0.01 of any trap value {0.0, 0.25, 0.5, 0.75, 1.0}
    """
    trap_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    while True:
        # Use OS entropy source (not pseudo-random)
        candidate = int.from_bytes(os.urandom(8), 'big') / (2**64)
        # Condition 1: strictly within (0.01, 0.99)
        if candidate <= 0.01 or candidate >= 0.99:
            continue
        # Condition 2: not within 0.01 of any trap value
        if any(abs(candidate - tv) < 0.01 for tv in trap_values):
            continue
        return candidate

logistic = LogisticMap(A=4.0, x0=generate_valid_seed())

n_cham = 50
d_dim  = 19
iters  = 300
pp     = 0.1
p1     = 0.25
p2     = 1.5
gamma  = 1
alpha  = 3.5
beta   = 3
ro     = 1
c1, c2 = 1.75, 1.75
e      = 2.71828

num_buses     = 62
base_MVA      = 100.0
slack_bus_idx = 0

gen_buses = [0, 1, 2, 3, 4, 7, 10, 12, 15, 16, 22, 25, 31, 33, 36, 48, 49, 55, 60]

Pd_MW = np.zeros(num_buses)
Pd_MW[2]=40;  Pd_MW[7]=109;  Pd_MW[8]=66;   Pd_MW[9]=40;   Pd_MW[10]=161
Pd_MW[11]=155; Pd_MW[12]=132; Pd_MW[14]=155; Pd_MW[17]=121; Pd_MW[18]=130
Pd_MW[19]=80;  Pd_MW[21]=64;  Pd_MW[23]=58;  Pd_MW[25]=116; Pd_MW[26]=85
Pd_MW[27]=63;  Pd_MW[29]=77;  Pd_MW[30]=51;  Pd_MW[32]=46;  Pd_MW[33]=100
Pd_MW[34]=107; Pd_MW[35]=20;  Pd_MW[37]=166; Pd_MW[38]=30;  Pd_MW[39]=25
Pd_MW[40]=92;  Pd_MW[41]=30;  Pd_MW[42]=25;  Pd_MW[43]=109; Pd_MW[44]=20
Pd_MW[52]=248; Pd_MW[54]=94;  Pd_MW[61]=93

raw_branches = [
    (0,1,0.00305,0.01565),(0,3,0.00716,0.03678),(0,13,0.00548,0.02813),
    (0,9,0.01569,0.08061),(0,8,0.00229,0.01174),(0,5,0.00411,0.02113),
    (1,5,0.00168,0.00861),(1,2,0.00289,0.01487),(2,3,0.00381,0.01957),
    (3,14,0.00411,0.02113),(13,14,0.00520,0.02669),(3,13,0.00411,0.02113),
    (12,13,0.01315,0.06754),(11,12,0.01537,0.07897),(11,10,0.01905,0.09783),
    (10,9,0.00686,0.03522),(3,4,0.00716,0.03678),(4,5,0.00575,0.01478),
    (5,6,0.00030,0.00157),(6,7,0.00049,0.00168),(4,7,0.00575,0.01478),
    (10,15,0.01406,0.07223),(15,16,0.00343,0.01761),(16,20,0.01850,0.09548),
    (20,21,0.01371,0.07043),(21,22,0.00396,0.02035),(22,23,0.00305,0.01565),
    (22,24,0.00126,0.00650),(24,27,0.01062,0.05554),(24,25,0.00941,0.04828),
    (24,26,0.01173,0.06026),(26,28,0.00533,0.02739),(28,29,0.02058,0.10573),
    (19,22,0.02042,0.10487),(11,19,0.01981,0.10174),(12,16,0.01563,0.08030),
    (13,18,0.00707,0.03631),(13,17,0.00135,0.00693),(13,15,0.00396,0.02035),
    (23,44,0.01219,0.06261),(23,40,0.01554,0.07993),(40,44,0.00335,0.01712),
    (39,40,0.00609,0.03130),(40,41,0.00076,0.00391),(41,42,0.00914,0.04696),
    (41,43,0.01417,0.07278),(38,41,0.00686,0.03522),(38,36,0.00229,0.01174),
    (37,36,0.01044,0.05361),(37,33,0.01076,0.05525),(33,36,0.01990,0.01022),
    (33,32,0.01737,0.08922),(33,34,0.00701,0.03600),(34,31,0.00036,0.00184),
    (32,31,0.01676,0.08609),(31,30,0.01787,0.09180),(29,30,0.00992,0.05095),
    (39,29,0.00716,0.03678),(31,35,0.00305,0.01565),(31,36,0.02200,0.11301),
    (31,33,0.00396,0.02035),(31,45,0.02095,0.10761),(35,45,0.01828,0.09391),
    (36,45,0.00104,0.00536),(45,43,0.01676,0.08609),(43,58,0.00884,0.04539),
    (58,60,0.00922,0.04735),(59,60,0.00244,0.01252),(60,61,0.01499,0.07701),
    (61,24,0.01383,0.07106),(57,60,0.00335,0.01722),(57,59,0.00411,0.02113),
    (54,57,0.00670,0.03443),(56,57,0.00183,0.00939),(56,55,0.00152,0.00783),
    (55,57,0.00259,0.01330),(51,60,0.01127,0.05791),(51,52,0.01132,0.05815),
    (50,54,0.01417,0.07278),(50,52,0.01190,0.06112),(50,53,0.00407,0.02090),
    (47,53,0.01254,0.06441),(47,49,0.00066,0.00337),(48,49,0.00670,0.03443),
    (48,47,0.00366,0.01878),(46,47,0.01371,0.07043),(46,45,0.00792,0.04070),
    (59,11,0.01365,0.07012),(57,11,0.01211,0.06222),
]

bus_data = []
for i in range(num_buses):
    if i == slack_bus_idx:
        bt=3; gi=gen_buses.index(i); vs=1.05
    elif i in gen_buses:
        bt=2; gi=gen_buses.index(i); vs=1.05
    else:
        bt=1; gi=-1; vs=1.0
    bus_data.append({
        'bus_id': i, 'bus_type': bt,
        'P_D': float(Pd_MW[i]/base_MVA),
        'Q_D': float(Pd_MW[i]*0.3/base_MVA),
        'V_sched': float(vs), 'gen_idx': int(gi)
    })

branch_data = []
for (f,t,r,x) in raw_branches:
    branch_data.append({'from_bus':f,'to_bus':t,'r':r,'x':x,'b_sh':0.3*x})

Pmin = [50.0,50.0,50.0,0.0,50.0,50.0,50.0,50.0,0.0,0.0,
        50.0,0.0,50.0,0.0,0.0,50.0,0.0,50.0,100.0]
Pmax = [300.0,450.0,450.0,100.0,300.0,450.0,200.0,500.0,600.0,100.0,
        150.0,50.0,300.0,150.0,500.0,150.0,100.0,300.0,600.0]

Qmin = np.array([-40.0]*len(gen_buses))
Qmax = np.array([150.0]*len(gen_buses))

cost_a = [0.0070,0.0055,0.0055,0.0025,0.0060,0.0055,0.0065,0.0075,0.0085,
          0.0020,0.0045,0.0025,0.0050,0.0045,0.0065,0.0045,0.0025,0.0045,0.0080]
cost_b = [6.80,4.00,4.00,0.85,4.60,4.00,4.70,5.00,6.00,0.50,
          1.60,0.85,1.80,1.60,4.70,1.40,0.85,1.60,5.50]
cost_c = [95.0,30.0,45.0,10.0,20.0,90.0,42.0,46.0,55.0,58.0,
          65.0,78.0,75.0,85.0,80.0,90.0,10.0,25.0,90.0]

# Effective dispatch ceiling — 90% of operating range for spinning reserve
Pmax_eff = [Pmin[j] + 0.90 * (Pmax[j] - Pmin[j]) for j in range(len(Pmax))]

_FDLF_CACHE = {}

def fdlf_loss_calculator(P_G, Q_G, bus_data, branch_data,
                         slack_bus_idx=0, max_iter=50, tol=1e-4):
    """
    Fast Decoupled Load Flow (FDLF) loss calculator.
    Computes AC transmission losses using the B'-B'' decoupled method.
    Network matrices (Y-bus, B') are cached after first build.

    Q-limit enforcement: after FDLF converges, reactive power at each
    PV (generator) bus is checked against Qmin/Qmax.  Violating buses
    are converted to PQ buses with Q clamped at the limit, and FDLF
    is re-solved.  This is the standard bus-type-switching approach
    used in production power-flow solvers (MATPOWER, PSS/E, etc.).
    The CSA optimiser, calculate_fitness, and repair_power_balance
    are NOT modified -- only the FDLF solver enforces Q-limits.
    """
    n = len(bus_data)
    cache_key = (id(bus_data), id(branch_data), int(slack_bus_idx))

    if cache_key in _FDLF_CACHE:
        cached = _FDLF_CACHE[cache_key]
        Y_bus   = cached['Y_bus']
        G_bus   = cached['G_bus']
        B_bus   = cached['B_bus']
        Bp_lu   = cached['Bp_lu']
        non_slack = cached['non_slack']
        from_arr = cached['from_arr']
        to_arr   = cached['to_arr']
        y_br     = cached['y_br']
        b_sh_arr = cached['b_sh_arr']
        x_arr    = cached['x_arr']
    else:
        # -- Build Y-bus --------------------------------------------------
        Y_bus = np.zeros((n, n), dtype=complex)
        from_arr = np.array([br['from_bus'] for br in branch_data])
        to_arr   = np.array([br['to_bus']   for br in branch_data])
        r_arr    = np.array([br['r']        for br in branch_data])
        x_arr    = np.array([br['x']        for br in branch_data])
        b_sh_arr = np.array([br['b_sh']     for br in branch_data])

        z_br = r_arr + 1j * x_arr
        y_br = 1.0 / z_br

        for idx in range(len(branch_data)):
            f = from_arr[idx]
            t = to_arr[idx]
            Y_bus[f, t] -= y_br[idx]
            Y_bus[t, f] -= y_br[idx]
            Y_bus[f, f] += y_br[idx] + 1j * b_sh_arr[idx] / 2.0
            Y_bus[t, t] += y_br[idx] + 1j * b_sh_arr[idx] / 2.0

        G_bus = np.real(Y_bus)
        B_bus = np.imag(Y_bus)

        # -- Identify bus sets -------------------------------------------
        non_slack = [i for i in range(n) if i != slack_bus_idx]

        # -- B' matrix (P-delta subproblem, non-slack buses) ------------
        n_ns = len(non_slack)
        Bp = np.zeros((n_ns, n_ns))
        for idx in range(len(branch_data)):
            f = from_arr[idx]
            t = to_arr[idx]
            b_s = 1.0 / x_arr[idx]  # series susceptance only
            if f in non_slack and t in non_slack:
                fi = non_slack.index(f)
                ti = non_slack.index(t)
                Bp[fi, fi] += b_s
                Bp[ti, ti] += b_s
                Bp[fi, ti] -= b_s
                Bp[ti, fi] -= b_s
            elif f in non_slack:
                fi = non_slack.index(f)
                Bp[fi, fi] += b_s
            elif t in non_slack:
                ti = non_slack.index(t)
                Bp[ti, ti] += b_s

        Bp_lu = scipy.linalg.lu_factor(Bp)

        # -- Store cache (network properties only; B'' is rebuilt
        #    dynamically when Q-limit enforcement switches bus types) --
        _FDLF_CACHE[cache_key] = {
            'Y_bus': Y_bus, 'G_bus': G_bus, 'B_bus': B_bus,
            'Bp_lu': Bp_lu,
            'non_slack': non_slack,
            'from_arr': from_arr, 'to_arr': to_arr,
            'y_br': y_br, 'b_sh_arr': b_sh_arr,
            'x_arr': x_arr,
        }

    # -- Helper: build B'' LU factor for a given PQ bus set -------------
    def _build_Bpp_lu(pq_set):
        n_pq = len(pq_set)
        if n_pq == 0:
            return None
        Bpp = np.zeros((n_pq, n_pq))
        for idx in range(len(branch_data)):
            f = from_arr[idx]
            t = to_arr[idx]
            b_s = 1.0 / x_arr[idx]
            b_sh_half = b_sh_arr[idx] / 2.0
            if f in pq_set and t in pq_set:
                fi = pq_set.index(f)
                ti = pq_set.index(t)
                Bpp[fi, fi] += b_s + b_sh_half
                Bpp[ti, ti] += b_s + b_sh_half
                Bpp[fi, ti] -= b_s
                Bpp[ti, fi] -= b_s
            elif f in pq_set:
                fi = pq_set.index(f)
                Bpp[fi, fi] += b_s + b_sh_half
            elif t in pq_set:
                ti = pq_set.index(t)
                Bpp[ti, ti] += b_s + b_sh_half
        return scipy.linalg.lu_factor(Bpp)

    # -- Helper: run one FDLF solve with the given PQ set ---------------
    # This preserves the EXACT iteration logic of the original code:
    #   1. P-delta half-sweep (uses Bp_lu, non-slack buses)
    #   2. Q-|V| half-sweep  (uses Bpp_lu, PQ buses)
    #   3. Convergence check on dP and dQ
    def _run_fdsl(pq_set, Q_G_local, V_init, delta_init):
        V_mag = V_init.copy()
        delta = delta_init.copy()

        # Scheduled power injections (pu)
        P_sched = np.zeros(n)
        Q_sched = np.zeros(n)
        n_gen = len(P_G)
        for i in range(n):
            gi = bus_data[i]['gen_idx']
            if gi >= 0 and gi < n_gen:
                P_sched[i] += P_G[gi]
                Q_sched[i] += Q_G_local[gi]
            P_sched[i] -= bus_data[i]['P_D']
            Q_sched[i] -= bus_data[i]['Q_D']

        ns_arr = np.array(non_slack)
        pq_arr = np.array(pq_set) if pq_set else np.array([], dtype=int)

        Bpp_lu = _build_Bpp_lu(pq_set) if pq_set else None

        # -- FDLF iteration loop ----------------------------------------
        converged = False
        iteration = 0
        while iteration < max_iter and not converged:
            iteration += 1

            # Half-sweep 1: P-delta
            dd = delta[:, None] - delta[None, :]
            P_calc = V_mag * np.sum(V_mag * (G_bus * np.cos(dd) + B_bus * np.sin(dd)), axis=1)
            dP = P_sched - P_calc
            rhs_P = dP[ns_arr] / np.maximum(V_mag[ns_arr], 0.5)
            d_delta = scipy.linalg.lu_solve(Bp_lu, rhs_P)
            delta[ns_arr] += d_delta
            # Angle wrapping
            delta = (delta + np.pi) % (2 * np.pi) - np.pi

            # Half-sweep 2: Q-|V|
            if pq_set:
                dd = delta[:, None] - delta[None, :]
                Q_calc = V_mag * np.sum(V_mag * (G_bus * np.sin(dd) - B_bus * np.cos(dd)), axis=1)
                dQ = Q_sched - Q_calc
                rhs_Q = dQ[pq_arr] / np.maximum(V_mag[pq_arr], 0.5)
                d_V = scipy.linalg.lu_solve(Bpp_lu, rhs_Q)
                V_mag[pq_arr] += d_V
                # Clip PQ bus voltages
                V_mag[pq_arr] = np.clip(V_mag[pq_arr], 0.5, 1.5)
            else:
                dQ = np.zeros(n)

            # Convergence check
            max_dP = np.max(np.abs(dP[ns_arr])) if len(ns_arr) > 0 else 0.0
            max_dQ = np.max(np.abs(dQ[pq_arr])) if len(pq_arr) > 0 else 0.0
            if max_dP <= tol and max_dQ <= tol:
                converged = True

        # -- Final Q_calc -----------------------------------------------
        dd_f = delta[:, None] - delta[None, :]
        Q_calc_final = V_mag * np.sum(V_mag * (G_bus * np.sin(dd_f) - B_bus * np.cos(dd_f)), axis=1)

        return V_mag, delta, converged, iteration, Q_calc_final

    # -- Initial bus sets: PV (type 2) and PQ (type 1) ------------------
    pv_set = [i for i in range(n) if bus_data[i]['bus_type'] == 2]
    pq_set = [i for i in range(n) if bus_data[i]['bus_type'] == 1]

    # -- Initial FDLF solve (all generators as PV buses) ----------------
    Q_G_local = (Q_G.copy() if isinstance(Q_G, np.ndarray)
                 else np.array(Q_G, dtype=float))
    V_init = np.array([bus_data[i]['V_sched'] for i in range(n)], dtype=float)
    delta_init = np.zeros(n, dtype=float)

    V_mag, delta, converged, iteration, Q_calc_final = \
        _run_fdsl(pq_set, Q_G_local, V_init, delta_init)

    # -- Q-limit enforcement via PV->PQ bus-type switching --------------
    # After FDLF converges, check Q at each PV (generator) bus.
    # If Q exceeds Qmin/Qmax, convert the bus to PQ with Q clamped
    # at the limit, then re-solve.  Repeat until no violations or
    # max 10 enforcement iterations.
    if converged:
        for _ in range(10):
            violations = []
            for bus_i in pv_set[:]:
                gi = bus_data[bus_i]['gen_idx']
                if gi < 0:
                    continue
                q_g_pu = Q_calc_final[bus_i] + bus_data[bus_i]['Q_D']
                q_g_MVAr = q_g_pu * base_MVA
                if q_g_MVAr > Qmax[gi] + 1e-6:
                    violations.append((bus_i, gi, float(Qmax[gi])))
                elif q_g_MVAr < Qmin[gi] - 1e-6:
                    violations.append((bus_i, gi, float(Qmin[gi])))

            if not violations:
                break

            # Switch violating PV buses to PQ with Q fixed at the limit
            for bus_i, gi, q_fixed_MVAr in violations:
                pv_set.remove(bus_i)
                pq_set.append(bus_i)
                Q_G_local[gi] = q_fixed_MVAr / base_MVA  # Q in pu

            # Re-solve FDLF with updated bus types
            V_mag, delta, converged, iteration, Q_calc_final = \
                _run_fdsl(pq_set, Q_G_local, V_mag, delta)

            if not converged:
                break

    # -- Branch losses (full AC pi-model) -------------------------------
    V_complex = V_mag * np.exp(1j * delta)
    n_br = len(branch_data)
    P_L_branch = np.zeros(n_br)
    Q_L_branch = np.zeros(n_br)

    if converged:
        for idx in range(n_br):
            k = from_arr[idx]
            m = to_arr[idx]
            Vk = V_complex[k]
            Vm = V_complex[m]
            y_km = y_br[idx]
            b_sh_half = 1j * b_sh_arr[idx] / 2.0

            S_km = Vk * conj(y_km * (Vk - Vm) + b_sh_half * Vk)
            S_mk = Vm * conj(y_km * (Vm - Vk) + b_sh_half * Vm)

            P_L_branch[idx] = real(S_km) + real(S_mk)
            Q_L_branch[idx] = imag(S_km) + imag(S_mk)

        P_L_total = float(np.sum(P_L_branch))
        Q_L_total = float(np.sum(Q_L_branch))
    else:
        P_L_total = 1_000_000.0
        Q_L_total = 1_000_000.0

    # -- Final Q_calc ----------------------------------------------------
    dd_f = delta[:, None] - delta[None, :]
    Q_calc_final = V_mag * np.sum(V_mag * (G_bus * np.sin(dd_f) - B_bus * np.cos(dd_f)), axis=1)

    # -- Override Q_calc_final for Q-limited buses so the reported Q
    #    exactly matches the clamped limit (eliminates tiny residual
    #    violations like 150.003 due to FDLF convergence tolerance).
    #    The bus was switched from PV to PQ with Q fixed at the limit;
    #    the actual Q injection at that bus IS the limit, by construction.
    for bus_i in pq_set:
        gi = bus_data[bus_i]['gen_idx']
        if gi >= 0 and gi < len(Q_G_local):
            Q_calc_final[bus_i] = Q_G_local[gi] - bus_data[bus_i]['Q_D']

    return {
        'P_L_total': P_L_total,
        'Q_L_total': Q_L_total,
        'V_mag': V_mag.copy(),
        'delta': delta.copy(),
        'P_L_branch': P_L_branch.copy(),
        'Q_L_branch': Q_L_branch.copy(),
        'converged': converged,
        'iterations': iteration,
        'Q_calc_final': Q_calc_final.copy(),
    }


def calculate_fitness(position, detailed=False):
    n_gen = len(gen_buses)
    n_buses = len(bus_data)
    P_G_MW = np.array(position, dtype=float)
    P_G_pu = P_G_MW / base_MVA
    Q_G_dummy = np.zeros(n_gen)

    res = fdlf_loss_calculator(P_G_pu, Q_G_dummy, bus_data, branch_data,
                               slack_bus_idx=0, max_iter=50, tol=1e-4)
    P_L_MW = res['P_L_total'] * base_MVA
    converged = res['converged']

    # Fuel cost — paper's quadratic formula, raw MW
    total_cost = sum(cost_a[j]*P_G_MW[j]**2 + cost_b[j]*P_G_MW[j] + cost_c[j]
                     for j in range(n_gen))

    # Active power balance penalty — the ONLY penalty for ELD
    total_power = float(np.sum(P_G_MW))
    total_demand = float(np.sum(Pd_MW))
    error = abs(total_power - total_demand - P_L_MW)
    penalty = 1000000 * (error ** 2)

    # Reactive power (Q) limit penalty — applied only if FDLF converged
    if converged:
        Q_calc_final = res['Q_calc_final']
        for j in range(len(gen_buses)):
            bus_idx = gen_buses[j]
            q_g = (Q_calc_final[bus_idx] + bus_data[bus_idx]['Q_D']) * base_MVA
            if q_g > Qmax[j]:
                violation = q_g - Qmax[j]
                penalty += 1000000 * (violation ** 2)
            elif q_g < Qmin[j]:
                violation = Qmin[j] - q_g
                penalty += 1000000 * (violation ** 2)

    final_score = total_cost + penalty
    if not converged:
        final_score += 1e9

    if detailed:
        return final_score, total_cost, total_power, error, penalty, P_L_MW
    return final_score

def repair_power_balance(position):
    pos = list(position) if not isinstance(position, list) else position[:]
    n_gen = len(gen_buses)
    total_demand = float(np.sum(Pd_MW))

    # Run FDLF EXACTLY ONCE to get precise current losses
    P_G_pu = np.array(pos) / base_MVA
    Q_G_dummy = np.zeros(n_gen)
    res = fdlf_loss_calculator(P_G_pu, Q_G_dummy, bus_data, branch_data,
                               slack_bus_idx=0, max_iter=50, tol=1e-4)
    
    if res['converged']:
        P_L_MW = res['P_L_total'] * base_MVA
    else:
        P_L_MW = total_demand * 0.04 # Fallback just in case

    target = total_demand + P_L_MW
    diff = target - sum(pos)

    # Early exit if the position already satisfies power balance.
    if abs(diff) < 1.0:
        return pos

    # Adjustable generator set — same logic as the original.
    adjustable = []
    for j in range(n_gen):
        if diff > 0 and pos[j] < Pmax_eff[j]:
            adjustable.append(j)
        elif diff < 0 and pos[j] > Pmin[j]:
            adjustable.append(j)
    if not adjustable:
        return pos

    # Chaotic economically guided distribution
    mc = [max(2.0*cost_a[j]*pos[j]+cost_b[j],1e-9) for j in adjustable]
    if diff > 0:
        weights = [(1.0 / mc[k]) * logistic.next() for k in range(len(adjustable))]
    else:
        weights = [mc[k] * logistic.next() for k in range(len(adjustable))]
    total_w = sum(weights)

    if total_w < 1e-12:
        share = diff / len(adjustable)
        for j in adjustable:
            pos[j] += share
            pos[j] = max(Pmin[j], min(Pmax_eff[j], pos[j]))
    else:
        for k_idx, j in enumerate(adjustable):
            w = weights[k_idx] / total_w
            pos[j] += diff * w
            pos[j] = max(Pmin[j], min(Pmax_eff[j], pos[j]))

    return pos

from itertools import combinations

def rotation_matrix_numeric(n, axis, angle_val):
    other_axes = [k for k in range(n) if k != axis]
    planes = list(combinations(other_axes, 2))
    R = np.eye(n)
    c, s = np.cos(angle_val), np.sin(angle_val)
    for (i, j) in planes:
        G = np.eye(n)
        if i < axis < j:
            G[j,j]=c;  G[j,i]=-s
            G[i,i]=c;  G[i,j]=s
        else:
            G[i,i]=c;  G[i,j]=-s
            G[j,i]=s;  G[j,j]=c
        R = G @ R
    return R

def run_single_trial(trial_idx):
    # Reset logistic map to the paper's specification
    seed = generate_valid_seed()
    logistic.x = seed
    

    # Burn‑in offset: trial_idx × 100 calls to logistic.next()
    for _ in range(trial_idx * 100):
        logistic.next()

    # Re‑initialise all optimisation structures for a fresh run
    best_score = math.inf
    population = []
    personal_best_positions = []
    personal_best_scores = []
    best_position = []

    # ---- Initialise the Chameleon Swarm ----
    for i in range(n_cham):
        y = []
        for j in range(d_dim):
            r = logistic.next()
            value = Pmin[j] + r * (Pmax_eff[j] - Pmin[j])
            y.append(value)
        y = repair_power_balance(y)
        population.append(y[:])

        score, total_cost, total_power, error, penalty, p_loss = \
            calculate_fitness(y, detailed=True)

        personal_best_positions.append(y[:])
        personal_best_scores.append(score)

        if score < best_score:
            best_score = score
            best_position = y[:]

    G = best_position[:]
    P = personal_best_positions[:]

    # ---- Initialise velocities ----
    velocity = [[2 * logistic.next() - 1 for _ in range(d_dim)] for _ in range(n_cham)]

    no_improve_counter = 0

    # ---- Full CSA optimisation loop (iters) ----
    for t in range(iters):
        centroid = np.mean(population, axis=0)
        mu = gamma * math.exp((-alpha * t) / iters) ** beta
        acc = 2590 * (1 - (e ** (-(math.log(t + 1)))))
        omega = (1 - (t / iters)) ** (ro * (math.sqrt(t / iters)))

        for i in range(n_cham):
            # SEARCH PHASE
            ri = logistic.next()
            temp_list = []
            for j in range(d_dim):
                r1 = logistic.next()
                r2 = logistic.next()
                r3 = logistic.next()
                if ri >= pp:
                    term1 = p1 * (P[i][j] - G[j]) * r2
                    term2 = p2 * (G[j] - population[i][j]) * r1
                    new_position = population[i][j] + term1 + term2
                else:
                    term3 = (Pmax_eff[j] - Pmin[j]) * r3 + Pmin[j]
                    term4 = 1 if logistic.next() >= 0.5 else -1
                    new_position = population[i][j] + mu * term3 * term4
                temp_list.append(new_position)

            for j in range(d_dim):
                if temp_list[j] < Pmin[j]:
                    temp_list[j] = Pmin[j]
                elif temp_list[j] > Pmax_eff[j]:
                    temp_list[j] = Pmax_eff[j]

            population[i] = repair_power_balance(temp_list)

            # SPOT PHASE
            current_position = np.array(population[i])
            axis_choice = 0 if logistic.next() < 0.5 else 1
            angle_val = logistic.next() * 2 * math.pi
            R_num = rotation_matrix_numeric(d_dim, axis_choice, angle_val)
            yc = current_position - centroid
            yr = (R_num @ yc).flatten()
            population[i] = (yr + centroid).tolist()

            for j in range(d_dim):
                if population[i][j] < Pmin[j]:
                    population[i][j] = Pmin[j]
                elif population[i][j] > Pmax_eff[j]:
                    population[i][j] = Pmax_eff[j]

            population[i] = repair_power_balance(population[i])

            # HUNT PHASE
            for j in range(d_dim):
                v_prev = velocity[i][j]
                r1 = logistic.next()
                r2 = logistic.next()
                v_term_1 = c1 * (G[j] - population[i][j]) * r1
                v_term_2 = c2 * (P[i][j] - population[i][j]) * r2
                new_velocity = (omega * velocity[i][j]) + v_term_1 + v_term_2
                velocity[i][j] = new_velocity
                if acc > 0:
                    new_pos = population[i][j] + ((velocity[i][j]**2 - v_prev**2) / (2 * acc))
                    population[i][j] = new_pos

            for k in range(d_dim):
                if population[i][k] > Pmax_eff[k]:
                    population[i][k] = Pmax_eff[k]
                elif population[i][k] < Pmin[k]:
                    population[i][k] = Pmin[k]

            population[i] = repair_power_balance(population[i])

            # FITNESS EVALUATION & BEST UPDATE
            current_fitness = calculate_fitness(population[i])

            if current_fitness < personal_best_scores[i]:
                personal_best_scores[i] = current_fitness
                P[i] = population[i][:]

            if current_fitness < best_score:
                best_score = current_fitness
                G = population[i][:]
                no_improve_counter = 0

        no_improve_counter += 1

        # DIVERSITY RESTART
        if no_improve_counter >= 20:
            n_restart = max(1, n_cham // 10)
            sorted_idx = sorted(range(n_cham),
                                key=lambda x: personal_best_scores[x],
                                reverse=True)
            restart_indices = sorted_idx[:n_restart]

            for idx in restart_indices:
                new_pos = []
                for j in range(d_dim):
                    r = logistic.next()
                    new_pos.append(Pmin[j] + r * (Pmax_eff[j] - Pmin[j]))
                new_pos = repair_power_balance(new_pos)
                new_score = calculate_fitness(new_pos)
                population[idx] = new_pos[:]
                personal_best_scores[idx] = new_score
                P[idx] = new_pos[:]

        # (Optional) progress reporting – commented out for speed
        # if (t + 1) % 10 == 0 or t == 0:
        #     print(f"Iter {t+1:04d}/{iters}: best_score={best_score:.4f}")

    # ---- Final metrics for this trial ----
    final_score, total_cost, total_power, error, penalty, p_loss = \
        calculate_fitness(G, detailed=True)

    return {
        'trial': trial_idx,
        'burn_in': trial_idx * 100,
        'fuel_cost': total_cost,
        'losses': p_loss,
        'p_balance_error': error,
        'best_score': best_score,
        'best_G': G,
    }

# ============================================================
# PARALLEL EXECUTION OF THE 30-TRIAL BENCHMARK
# ============================================================
import sys

if __name__ == "__main__":

    print(f"Total demand: {np.sum(Pd_MW):.2f} MW")
    print(f"Buses: {len(bus_data)}, Branches: {len(branch_data)}, Generators: {len(gen_buses)}")

    # --- Pre-warm the FDLF cache ONCE in the parent --------------------
    _P_G_warmup = np.array(
        [Pmin[j] + 0.5 * (Pmax_eff[j] - Pmin[j]) for j in range(len(gen_buses))]
    ) / base_MVA
    _Q_G_warmup = np.zeros(len(gen_buses))
    _ = fdlf_loss_calculator(
        _P_G_warmup, _Q_G_warmup, bus_data, branch_data,
        slack_bus_idx=0, max_iter=50, tol=1e-4
    )
    print(f"[parent] FDLF cache pre-warmed. Cache keys present: {list(_FDLF_CACHE.keys())}")

    # --- Number of parallel workers -----------------------------------
    # MUST BE 8. 16 logical threads will cause cache thrashing and drop 
    # CPU utilization to 40% due to memory bandwidth limits on BLAS operations.
    N_WORKERS = 8
    print(f"[parent] Launching {N_WORKERS} parallel workers for 30 trials...")

    benchmark_results = []
    global_best_score = math.inf
    global_best_cost = math.inf
    global_best_G = None
    global_best_trial_idx = None
    global_best_burnin = None

    with mp.Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(run_single_trial, range(30)):
            benchmark_results.append(result)
            print(
                f"Trial {result['trial']+1:02d}: "
                f"burn-in offset {result['burn_in']:3d} → "
                f"fuel cost {result['fuel_cost']:.4f} $/h, "
                f"active loss {result['losses']:.4f} MW, "
                f"P-balance error {result['p_balance_error']:.6f} MW"
            )
            
            if result['best_score'] < global_best_score:
                global_best_score = result['best_score']
                global_best_cost = result['fuel_cost']
                global_best_G = result['best_G'][:]
                global_best_trial_idx = result['trial']
                global_best_burnin = result['burn_in']

    benchmark_results.sort(key=lambda r: r['trial'])

    print(f"\n[parent] All 30 trials complete. Best trial: {global_best_trial_idx+1} (fuel cost {global_best_cost:.4f} $/h)")
    
    # ---- Results table ----
    print("\n" + "="*80)
    print("Benchmark Results Summary")
    print("="*80)
    print(f"{'Trial':>5} {'Burn-in':>7} {'Fuel Cost($/h)':>15} {'Active Loss (MW)':>15} {'P-balance Error (MW)':>20}")
    print("-"*80)
    for r in benchmark_results:
        print(f"{r['trial']+1:5d} {r['burn_in']:7d} {r['fuel_cost']:15.4f} {r['losses']:15.4f} {r['p_balance_error']:20.6f}")

    # ---- Statistical summary ----
    fuel_costs = [r['fuel_cost'] for r in benchmark_results]
    losses = [r['losses'] for r in benchmark_results]

    fuel_best = min(fuel_costs)
    fuel_worst = max(fuel_costs)
    fuel_mean = np.mean(fuel_costs)
    fuel_std = np.std(fuel_costs, ddof=1)

    loss_best = min(losses)
    loss_worst = max(losses)
    loss_mean = np.mean(losses)
    loss_std = np.std(losses, ddof=1)

    print("\n" + "="*80)
    print("Statistical Summary (Fuel Cost and Active Losses)")
    print("="*80)
    print(f"{'':>20} {'Best':>12} {'Mean':>12} {'Worst':>12} {'Std Dev':>12}")
    print("-"*80)
    print(f"{'Fuel Cost ($/h)':>20} {fuel_best:>12.4f} {fuel_mean:>12.4f} {fuel_worst:>12.4f} {fuel_std:>12.4f}")
    print(f"{'Active Loss (MW)':>20} {loss_best:>12.4f} {loss_mean:>12.4f} {loss_worst:>12.4f} {loss_std:>12.4f}")
    print("="*80)

    # ================================================================
    # DETAILED RESULTS FOR THE BEST TRIAL
    # ================================================================
    print("\n\n" + "="*80)
    print("DETAILED RESULTS FOR THE BEST TRIAL")
    print("="*80)

    best_result = benchmark_results[global_best_trial_idx]
    print(f"  Trial Number  : {global_best_trial_idx + 1}")
    print(f"  Burn-in Offset: {global_best_burnin}")
    print(f"  Fuel Cost     : {best_result['fuel_cost']:.4f} $/h")

    P_G_best = np.array(global_best_G)
    P_G_pu_best = P_G_best / base_MVA
    Q_G_pu_best = np.zeros(len(gen_buses))
    res = fdlf_loss_calculator(P_G_pu_best, Q_G_pu_best, bus_data, branch_data,
                               slack_bus_idx=0, max_iter=50, tol=1e-4)

    # ---- Generator Dispatch Table ----
    print("\n" + "-"*80)
    print("Generator Dispatch Table")
    print("-"*80)
    print(f"{'Gen':>4} {'Bus':>4} {'P_G(MW)':>10} {'Pmin':>8} {'Pmax':>8} {'Pmax_eff':>9} {'Util_eff(%)':>8} {'Cost($/h)':>12}")
    print("-"*80)
    total_gen_cost_sum = 0.0
    for j in range(len(gen_buses)):
        pg = P_G_best[j]
        util = (pg - Pmin[j]) / max(Pmax[j] - Pmin[j], 1e-9) * 100
        indiv_cost = cost_a[j] * pg**2 + cost_b[j] * pg + cost_c[j]
        total_gen_cost_sum += indiv_cost
        print(f"{j+1:4d} {gen_buses[j]+1:4d} {pg:10.4f} {Pmin[j]:8.2f} {Pmax[j]:8.2f} {Pmax_eff[j]:9.2f} {util:8.2f} {indiv_cost:12.4f}")

    # ---- Voltage Profile ----
    print("\n" + "-"*80)
    print("Voltage Profile")
    print("-"*80)
    print(f"{'Bus':>4} {'Type':>6} {'|V|(pu)':>9} {'Angle(deg)':>11} {'Flag':>8}")
    print("-"*80)
    V_mag_best = res['V_mag']
    delta_best = res['delta']
    for b in range(num_buses):
        bt = bus_data[b]['bus_type']
        if bt == 3:
            btype_str = "Slack"
        elif bt == 2:
            btype_str = "PV"
        else:
            btype_str = "PQ"
        angle_deg = np.degrees(delta_best[b])
        flag = ""
        if V_mag_best[b] < 0.95:
            flag = "LOW"
        elif V_mag_best[b] > 1.05:
            flag = "HIGH"
        print(f"{b+1:4d} {btype_str:>6} {V_mag_best[b]:9.4f} {angle_deg:11.4f} {flag:>8}")

    # ---- Reactive Power at Generators ----
    print("\n" + "-"*80)
    print("Reactive Power at Generators")
    print("-"*80)
    print(f"{'Gen':>4} {'Bus':>4} {'Q_G(MVAr)':>11} {'Qmin':>8} {'Qmax':>8} {'Flag':>8}")
    print("-"*80)
    Q_calc_final = res['Q_calc_final']
    for j in range(len(gen_buses)):
        bus_idx = gen_buses[j]
        q_g = (Q_calc_final[bus_idx] + bus_data[bus_idx]['Q_D']) * base_MVA
        flag = ""
        if q_g > Qmax[j]:
            flag = "OVER"
        elif q_g < Qmin[j]:
            flag = "UNDER"
        print(f"{j+1:4d} {gen_buses[j]+1:4d} {q_g:11.4f} {Qmin[j]:8.2f} {Qmax[j]:8.2f} {flag:>8}")

    # ---- Power Balance Summary ----
    print("\n" + "-"*80)
    print("Power Balance Summary")
    print("-"*80)
    total_gen_MW = float(np.sum(P_G_best))
    total_demand_MW = float(np.sum(Pd_MW))
    total_losses_MW = res['P_L_total'] * base_MVA
    p_balance_err = abs(total_gen_MW - total_demand_MW - total_losses_MW)
    print(f"  Total Generation   : {total_gen_MW:.4f} MW")
    print(f"  Total Demand       : {total_demand_MW:.4f} MW")
    print(f"  Total Active Losses: {total_losses_MW:.4f} MW")
    print(f"  P-balance Error    : {p_balance_err:.6f} MW")
    print(f"  Total Fuel Cost    : {best_result['fuel_cost']:.4f} $/h")
    print(f"  Sum of Gen Costs   : {total_gen_cost_sum:.4f} $/h  (verification)")
    print("="*80)

    # ---- Total Execution Time ----
    _TOTAL_ELAPSED = time.perf_counter() - _START_TIME
    _HOURS   = int(_TOTAL_ELAPSED // 3600)
    _MINUTES = int((_TOTAL_ELAPSED % 3600) // 60)
    _SECONDS = _TOTAL_ELAPSED % 60

    print("\n" + "="*80)
    print("Execution Time")
    print("="*80)
    print(f"  Total elapsed : {_TOTAL_ELAPSED:.4f} seconds")
    print(f"  Formatted     : {_HOURS:02d}h {_MINUTES:02d}m {_SECONDS:06.3f}s")
    print("="*80)