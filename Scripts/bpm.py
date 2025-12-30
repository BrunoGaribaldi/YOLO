import numpy as np
import re

def load_points(path):
    frames, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\((\d+),\s*([0-9.]+)\)", line)
            if m:
                frames.append(int(m.group(1)))
                ys.append(float(m.group(2)))
    return np.array(frames), np.array(ys)

def interpolate_signal(frames, ys):
    n = frames.max() + 1
    y = np.full(n, np.nan, dtype=float)
    y[frames] = ys
    idx = np.arange(n)
    good = np.isfinite(y)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y

def smooth_ma(y, win=11):
    win = max(1, int(win))
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")

def find_extrema(y):
    dy = np.diff(y)
    s = np.sign(dy)
    s[s == 0] = 1e-6
    max_idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1
    min_idx = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1
    return max_idx, min_idx

def group_close(idxs, y, merge_dist, mode):
    if len(idxs) == 0:
        return idxs
    idxs = np.sort(idxs)
    groups = [[idxs[0]]]
    for i in idxs[1:]:
        if i - groups[-1][-1] <= merge_dist:
            groups[-1].append(i)
        else:
            groups.append([i])

    out = []
    for g in groups:
        if mode == "max":
            out.append(max(g, key=lambda j: y[j]))
        else:
            out.append(min(g, key=lambda j: y[j]))
    return np.array(out, dtype=int)

def alternating_extrema(y, fps, bpm_min=2, bpm_max=20, smooth_win=11):
    y_s = smooth_ma(y, win=smooth_win)
    max_idx, min_idx = find_extrema(y_s)

    # merge de extremos cercanos (mesetas): usar una fracción del período mínimo
    min_period = int(fps * 60 / bpm_max)
    merge_dist = max(1, int(min_period * 0.15))

    max_idx = group_close(max_idx, y_s, merge_dist, mode="max")
    min_idx = group_close(min_idx, y_s, merge_dist, mode="min")

    events = [(i, "max") for i in max_idx] + [(i, "min") for i in min_idx]
    events.sort(key=lambda x: x[0])
    if not events:
        return [], y_s

    # fuerza alternancia: si hay dos seguidos del mismo tipo, conserva el más extremo
    cleaned = [events[0]]
    for i, t in events[1:]:
        pi, pt = cleaned[-1]
        if t != pt:
            cleaned.append((i, t))
        else:
            if t == "max" and y_s[i] > y_s[pi]:
                cleaned[-1] = (i, t)
            if t == "min" and y_s[i] < y_s[pi]:
                cleaned[-1] = (i, t)

    return cleaned, y_s

def bpm_cycle(points_frames, points_y, fps, bpm_min=2, bpm_max=20, smooth_win=11):
    y = interpolate_signal(points_frames, points_y)
    events, _ = alternating_extrema(y, fps, bpm_min=bpm_min, bpm_max=bpm_max, smooth_win=smooth_win)

    # período de ciclo completo: cada 2 eventos del mismo tipo
    periods = []
    for k in range(2, len(events)):
        i, t = events[k]
        j, tt = events[k - 2]
        if t == tt:
            periods.append((i - j) / fps)

    if len(periods) < 1:
        return None, {"reason": "no hay ciclos completos suficientes", "events": events[:10]}

    T = np.median(periods)
    bpm = 60 / T
    return bpm, {"events_sample": events[:10], "median_period_s": T, "n_periods": len(periods)}

