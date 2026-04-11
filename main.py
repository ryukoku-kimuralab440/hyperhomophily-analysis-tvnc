import os, sys
import json
import pickle
import numpy as np
from scipy.special import gammaln
from itertools import combinations
from collections import defaultdict


if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

CATEGORIES = [
    "C1",
    "C2",
    "C3",
    "Other"
]
CATEGORIES_ID = {cat:i for i, cat in enumerate(CATEGORIES)}
CATEGORIES_ID_INV = {i:cat for cat, i in CATEGORIES_ID.items()}
HL_CATEGORIES = [f"{cat}_{hl}" for hl in ("L","H") for cat in CATEGORIES]
HL_CATEGORIES_ID = {cat_hl:i for i, cat_hl in enumerate(HL_CATEGORIES)}
HL_CATEGORIES_ID_INV = {i:cat_hl for i, cat_hl in enumerate(HL_CATEGORIES)}
VER = 1


def load_json(path="./data.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(path="./data.pickle"):
    with open(path, 'rb') as f:
        return pickle.load(f)


def generate_configs(k, C):
    """ stars and bars method """
    config_idr = dict()
    comb = np.array(list(combinations(range(1, k + C), C - 1)), dtype=int)
    configs = np.zeros((comb.shape[0], C), dtype=int)
    for i in range(comb.shape[0]):
        bars = np.concatenate(([0], comb[i], [k + C]))
        pattern = np.diff(bars) - 1
        configs[i, :] = pattern
        config_idr[tuple(map(int, pattern))] = i
    return configs, config_idr


def np_savetxt(d, outpath, labels=None, header=[], delimiter=",", comments="", dim=2):
    if dim == 1:
        d = d[:, None]
    if labels is not None:
        print(labels)
        col0_labels = np.array(labels)
        row0_header = ",".join(header) if len(header) > 0 else ""
        print(f"labels:{col0_labels.shape} d:{d.shape}")
        d = np.column_stack((col0_labels, d))
    np.savetxt(outpath, d, delimiter=delimiter, header=row0_header, comments="", fmt="%s")


def gen_filtered_data(hyperedges, k=3, M=50, C=2, C0=4):
    print(f"categories:{CATEGORIES}")
    print(f"hl_categories:{HL_CATEGORIES}")
    cat_check = set()
    configs, config_idr = generate_configs(k, C)
    config_count = np.zeros((M, configs.shape[0]), dtype=int)
    #
    cnt_m = defaultdict(int)
    k_unique_nodes = set()
    xs = defaultdict(set)
    xms = defaultdict(lambda: defaultdict(set))
    d_m = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    print(f"hyperedge:{len(hyperedges):,}")
    num_hyperedges = 0
    for user_id, edges in hyperedges.items():
        for edge in edges:
            m = edge['week']
            if len(edge['nodes']) == k:
                num_hyperedges += 1
                cnt_m[m] += 1
                nodes = []
                for node_info in edge['nodes']:
                    try:
                        cat, hl, node_id = node_info[0], node_info[1], node_info[2]
                    except IndexError:
                        print(f"IndexError: node_info:{node_info}")
                        sys.exit()
                    cat_check.add(cat)
                    nodes.append([f"{cat}", f"{cat}_{hl}", node_id])
                hl_cat_nodes = []
                for cat, cat_hl, node_id in nodes:
                    k_unique_nodes.add(node_id)
                    xs[cat].add(node_id)
                    xms[cat_hl][m].add(node_id)
                    hl_cat_nodes.append(cat_hl)
                config_type = []
                for cat_hl in HL_CATEGORIES:
                    t = hl_cat_nodes.count(cat_hl)
                    d_m[cat_hl][m][t] += 1
                    config_type.append(t)
                try:
                    config_count[m, config_idr[tuple(config_type)]] += 1
                except KeyError:
                    print("KeyError")
                    print(f"hl_cat_nodes:{hl_cat_nodes}")
                    print(f"config_type:{config_type}")
                    sys.exit()
    return configs, config_count, k_unique_nodes, num_hyperedges


def calc_single(configs, config_count, Z_count, Z_base_count, C0=2, C=4, n=1, k=3, M=50):
    print(f"(calc_single) n:{n}, k:{k}, M:{M}, C0:{C0}, C:{C}")
    """
     calc single class homophily
    """
    a1 = np.zeros((k, C), dtype=float)
    a0 = np.zeros(C, dtype=float)
    a_single = np.zeros((k, C), dtype=float)
    b_single = np.zeros((k, C), dtype=float)
    h_single = np.zeros((k, C), dtype=float)
    h_single_plus = np.zeros(C, dtype=float)
    h_single_bar = np.zeros(C, dtype=float)
    # a1, a0
    for c in range(C):
        for t in range(1, k + 1):
            mask_single = (configs[:, c] == t)
            a1[t - 1, c] = np.sum([t * config_count[m, mask_single].sum() for m in range(M)])
        a0[c] = a1[:, c].sum()
    # b_single
    for c in range(C):
        for t in range(1, k + 1):
            val_b = 0.0
            for m in range(M):
                zc = Z_count[m, c]
                if zc < t or (n - zc) < (k - t):
                    if False:
                        raise ValueError(f"zc:{zc} < t:{t} || (n:{n} - zc:{zc}) < (k:{k} - t:{t}) c:{HL_CATEGORIES_ID_INV[c]}")
                    else:
                        continue
                log_val_b = (
                    (gammaln(zc) - gammaln(t) - gammaln(zc - t + 1))
                    + (gammaln(n - zc + 1) - gammaln(k - t + 1) - gammaln(n - zc - (k - t) + 1))
                    - (gammaln(n) - gammaln(k) - gammaln(n - k + 1))
                )
                val_b += np.exp(log_val_b)
            b_single[t - 1, c] = val_b / M
    # h_single
    for c in range(C):
        for t in range(1, k + 1):
            a_single[t - 1, c] = a1[t - 1, c] / a0[c] if a0[c] > 0 else np.nan
            h_single[t - 1, c] = a_single[t - 1, c] / b_single[t - 1, c]
    # Majority homophily
    t_mjr = int(np.ceil(k / 2))
    for c in range(C):
        h_single_plus[c] = a_single[t_mjr - 1 : k, c].sum() / b_single[t_mjr - 1 : k, c].sum()
    # Perfect homophily (t=k)
    for c in range(C):
        h_single_bar[c] = h_single[k - 1, c]

    # ============================================================
    # base
    # ============================================================
    a1_base = np.zeros((k, C0), dtype=float)
    a0_base = np.zeros(C0, dtype=float)
    a_base = np.zeros((k, C0), dtype=float)
    b_base = np.zeros((k, C0), dtype=float)     # 
    h_base = np.zeros((k, C0), dtype=float)     # single h0 for C0
    h_base_plus = np.zeros(C0, dtype=float)     # single h0+ for C0
    h_base_bar = np.zeros(C0, dtype=float)      # single bar h0 for C0
    for c0 in range(C0):
        r_base_all = configs[:, c0] + configs[:, c0 + C0]
        for t in range(1, k + 1):
            mask_base = (r_base_all == t)
            a1_base[t - 1, c0] = np.sum([t * config_count[m, mask_base].sum() for m in range(M)])
        a0_base[c0] = a1_base[:, c0].sum()
    for c0 in range(C0):
        for t in range(1, k + 1):
            zc0 = Z_base_count[c0]
            if zc0 < t or (n - zc0) < (k - t):
                if False:
                    raise ValueError("Base class count infeasible")
                else:
                    b_base[t - 1, c0] = 0.0
                    continue
            log_val_b = (
                (gammaln(zc0) - gammaln(t) - gammaln(zc0 - t + 1))
                + (gammaln(n - zc0 + 1) - gammaln(k - t + 1) - gammaln(n - zc0 - (k - t) + 1))
                - (gammaln(n) - gammaln(k) - gammaln(n - k + 1))
            )
            b_base[t - 1, c0] = np.exp(log_val_b)
    for c0 in range(C0):
        for t in range(1, k + 1):
            a_base[t - 1, c0] = a1_base[t - 1, c0] / a0_base[c0] if a0_base[c0] > 0 else np.nan
            h_base[t - 1, c0] = a_base[t - 1, c0] / b_base[t - 1, c0]
    # Static Majority homophily h^(0)_+(X)
    t_mjr = int(np.ceil(k / 2))
    for c in range(C0):
        h_base_plus[c] = a_base[t_mjr - 1 : k, c].sum() / b_base[t_mjr - 1 : k, c].sum()
    # Static Perfect homophily (t=k) \bar{h}^(0)(X)
    for c in range(C0):
        h_base_bar[c] = h_base[k - 1, c]

    return h_base, h_single, h_base_plus, h_single_plus, h_base_bar, h_single_bar


def calc_pair(configs, config_count, Z_count, C0=2, C=4, n=1, k=3, M=50):
    print(f"(calc_pair) n:{n}, k:{k}, M:{M}, C0:{C0}, C:{C}")
    # ============================================================
    # Majority homophily for category pairs (extended class)
    # ============================================================
    hh_plus = np.full((C, C), np.nan, dtype=float)
    aa_plus = np.full((C, C), np.nan, dtype=float)
    val_aa1 = np.zeros((C, C), dtype=float)
    val_aa0 = np.zeros((C, C), dtype=float)

    for c1 in range(C - 1):
        for c2 in range(c1 + 1, C):
            mask_plus = (
                (configs[:, c1] + configs[:, c2] > k / 2)
                & (configs[:, c1] > 0)
                & (configs[:, c2] > 0)
            )
            id_plus = np.where(mask_plus)[0]
            t1_plus = configs[id_plus, c1]
            t2_plus = configs[id_plus, c2]

            mask_0 = (configs[:, c1] > 0) & (configs[:, c2] > 0)
            id_0 = np.where(mask_0)[0]
            t1_0 = configs[id_0, c1]
            t2_0 = configs[id_0, c2]

            for m in range(M):
                val_aa1[c1, c2] += np.sum(t1_plus * t2_plus * config_count[m, id_plus])
                val_aa0[c1, c2] += np.sum(t1_0 * t2_0 * config_count[m, id_0])

            aa_plus[c1, c2] = val_aa1[c1, c2] / val_aa0[c1, c2] if val_aa0[c1, c2] > 0 else np.nan

    bb_plus = np.full((C, C), np.nan, dtype=float)
    s1, s2 = np.meshgrid(np.arange(1, k), np.arange(1, k), indexing="ij")
    mask_b = (s1 + s2 > k / 2) & (s1 + s2 <= k)
    s_pairs = np.column_stack([s1[mask_b], s2[mask_b]])

    for c1 in range(C - 1):
        for c2 in range(c1 + 1, C):
            val_bb_plus = 0.0
            for m in range(M):
                zc1 = Z_count[m, c1]
                zc2 = Z_count[m, c2]
                zc_other = n - zc1 - zc2
                for ss1, ss2 in s_pairs:
                    log_val_bb = (
                        (gammaln(zc1) - gammaln(ss1) - gammaln(zc1 - ss1 + 1))
                        + (gammaln(zc2) - gammaln(ss2) - gammaln(zc2 - ss2 + 1))
                        + (gammaln(zc_other + 1) - gammaln(k - ss1 - ss2 + 1) - gammaln(zc_other - (k - ss1 - ss2) + 1))
                        - (gammaln(n - 1) - gammaln(k - 1) - gammaln(n - k + 1))
                    )
                    val_bb_plus += np.exp(log_val_bb)
            bb_plus[c1, c2] = val_bb_plus / M
            hh_plus[c1, c2] = aa_plus[c1, c2] / bb_plus[c1, c2]

    # ===============================================================
    # Perfect homophily analysis for category pairs (extended class)
    # ===============================================================
    hh_bar = np.full((C, C), np.nan, dtype=float)
    aa_bar = np.full((C, C), np.nan, dtype=float)
    val_aa1_bar = np.zeros((C, C), dtype=float)
    val_aa0_bar = np.zeros((C, C), dtype=float)

    for c1 in range(C - 1):
        for c2 in range(c1 + 1, C):
            mask_bar = (
                (configs[:, c1] + configs[:, c2] == k)
                & (configs[:, c1] > 0)
                & (configs[:, c2] > 0)
            )
            id_bar = np.where(mask_bar)[0]
            t1_bar = configs[id_bar, c1]
            t2_bar = configs[id_bar, c2]

            mask_0 = (configs[:, c1] > 0) & (configs[:, c2] > 0)
            id_0 = np.where(mask_0)[0]
            t1_0 = configs[id_0, c1]
            t2_0 = configs[id_0, c2]

            for m in range(M):
                val_aa1_bar[c1, c2] += np.sum(t1_bar * t2_bar * config_count[m, id_bar])
                val_aa0_bar[c1, c2] += np.sum(t1_0 * t2_0 * config_count[m, id_0])

            aa_bar[c1, c2] = val_aa1_bar[c1, c2] / val_aa0_bar[c1, c2] if val_aa0_bar[c1, c2] > 0 else np.nan

    bb_bar = np.full((C, C), np.nan, dtype=float)
    u1, u2 = np.meshgrid(np.arange(1, k), np.arange(1, k), indexing="ij")
    mask_u = (u1 + u2 == k)
    u_pairs = np.column_stack([u1[mask_u], u2[mask_u]])

    for c1 in range(C - 1):
        for c2 in range(c1 + 1, C):
            val_bb_bar = 0.0
            for m in range(M):
                zc1 = Z_count[m, c1]
                zc2 = Z_count[m, c2]
                zc_other = n - zc1 - zc2
                for ss1, ss2 in u_pairs:
                    log_val_bb = (
                        (gammaln(zc1) - gammaln(ss1) - gammaln(zc1 - ss1 + 1))
                        + (gammaln(zc2) - gammaln(ss2) - gammaln(zc2 - ss2 + 1))
                        + (gammaln(zc_other + 1) - gammaln(k - ss1 - ss2 + 1) - gammaln(zc_other - (k - ss1 - ss2) + 1))
                        - (gammaln(n - 1) - gammaln(k - 1) - gammaln(n - k + 1))
                    )
                    val_bb_bar += np.exp(log_val_bb)
            bb_bar[c1, c2] = val_bb_bar / M
            hh_bar[c1, c2] = aa_bar[c1, c2] / bb_bar[c1, c2]

    return hh_plus, hh_bar


def gen_z_count(k_unique_nodes, node_labels, C0=2, C=4, M=52):
    Z_count = np.zeros((M, C), dtype=int)
    Z_base_count = np.zeros(C0, dtype=int)
    for node in k_unique_nodes:
        cat = node_labels[node]["category"]
        c0 = CATEGORIES_ID[cat]
        for m in range(M):
            try:
                hl = node_labels[node]["weekly_labels"][str(m+1)]
            except KeyError:
                print(f"KeyError:{node_labels[node]['weekly_labels']}")
                sys.exit()
            c = HL_CATEGORIES_ID[f"{cat}_{hl}"]
            Z_count[m, c] += 1
        Z_base_count[c0] += 1
    n = len(k_unique_nodes)
    return n, Z_count, Z_base_count


def main():
    OUTPUT_DIR = './result'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DATASET = "sample"
    data_path = f"./{DATASET}_data"
    C0 = len(CATEGORIES)    # # of base classes
    C = 2 * C0              # # of extended classes: High and Low activities
    M = 52                  # # of time units: weeks
    k_targets = [5]
    for k in k_targets:   # size of k-uniform hyperedge
        HYPEREDGE_JSON = f'{data_path}/user_hyperedges_k{k}.pickle'
        NODE_LABELS_JSON = f'{data_path}/pois_k{k}.pickle'
        user_hyperedges = load_pickle(HYPEREDGE_JSON)
        node_labels = load_pickle(NODE_LABELS_JSON)
        print(f"Target size k: {k}")
        configs, config_count, k_unique_nodes, num_hyperedges = gen_filtered_data(user_hyperedges, k=k, M=M, C0=C0, C=C)
        n, Z_count, Z_base_count = gen_z_count(k_unique_nodes, node_labels, C0=C0, C=C, M=M)
        print(f"n:{n:,}")
        h_base, h_single, h_base_plus, h_single_plus, h_base_bar, h_single_bar = \
            calc_single(configs, config_count, Z_count, Z_base_count, C0=C0, C=C, n=n, k=k, M=M)
        hh_plus, hh_bar = \
            calc_pair(configs, config_count, Z_count, C0=C0, C=C, n=n, k=k, M=M)
        header = ["class"] + [f"{i+1}" for i in range(k)]
        if True:
            print(f"Static Homophily Index for affinity type and Base-Category:\n{h_base}")
            fn = f"{OUTPUT_DIR}/{DATASET}_single_ht0_k{k}.csv"
            np_savetxt(h_base.T, fn, header=header, labels=[cat for cat in CATEGORIES])
        if True:
            print(f"\nHomophily Index for affinity type and Category:\n{h_single}")
            fn = f"{OUTPUT_DIR}/{DATASET}_single_ht_HL_k{k}.csv"
            np_savetxt(h_single.T, fn, header=header, labels=[cat for cat in HL_CATEGORIES])
        if True:
            print(f"\nStatic Majority Homophily Index for Base-Category :\n{h_base_plus}")
            fn = f"{OUTPUT_DIR}/{DATASET}_single_h0plus_k{k}.csv"
            np_savetxt(h_base_plus, fn, header=[], labels=[cat for cat in CATEGORIES], dim=1)
        if True:
            print(f"\nStatic Perfect Homophily Index for Base-Category :\n{h_base_bar}")
            fn = f"{OUTPUT_DIR}/{DATASET}_single_h0bar_k{k}.csv"
            np_savetxt(h_base_bar, fn, header=[], labels=[cat for cat in CATEGORIES], dim=1)
        if True:
            print(f"\nMajority Homophily Index for Category :\n{h_single_plus}")
            fn = f"{OUTPUT_DIR}/{DATASET}_single_hplus_HL_k{k}.csv"
            np_savetxt(h_single_plus, fn, header=[], labels=[cat for cat in HL_CATEGORIES], dim=1)
        if True:
            print(f"\nPerfect Homophily Index for Category:\n{h_single_bar}")
            fn = f"{OUTPUT_DIR}/{DATASET}_single_hbar_HL_k{k}.csv"
            np_savetxt(h_single_bar, fn, header=[], labels=[cat for cat in HL_CATEGORIES], dim=1)
        pair_header = ["class"] + [cat for cat in HL_CATEGORIES]
        if True:
            print(f"\nMajority Homophily Index for Category Pair:\n{hh_plus}")
            fn = f"{OUTPUT_DIR}/{DATASET}_pair_hplus_k{k}.csv"
            np_savetxt(hh_plus, fn, header=pair_header, labels=[cat for cat in HL_CATEGORIES])
        if True:
            print(f"\nPerfect Homophily Index for Category Pair:\n{hh_bar}")
            fn = f"{OUTPUT_DIR}/{DATASET}_pair_hbar_k{k}.csv"
            np_savetxt(hh_bar, fn, header=pair_header, labels=[cat for cat in HL_CATEGORIES])
        if True:
            for c0, cat in CATEGORIES_ID_INV.items():
                print(f"{c0+1:02d}:{cat}\t{Z_base_count[c0]:,}")
            print(f"n:{n:,}")
            print(f"# of hyperedges with k={k}: {num_hyperedges:,}")


if __name__ == "__main__":
    main()
