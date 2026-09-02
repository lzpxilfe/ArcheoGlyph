# -*- coding: utf-8 -*-
"""
Class-break computation for graduated symbol sizing.

Pure Python, no QGIS dependency, so it can be unit tested directly.
Mode codes match the main dialog's size-mode combo:
    1 = natural breaks (Jenks), 2 = equal interval, 3 = quantile.
"""

import math

MODE_FIXED = 0
MODE_NATURAL_BREAKS = 1
MODE_EQUAL_INTERVAL = 2
MODE_QUANTILE = 3


def compute_breaks(values, num_classes, size_mode):
    """Return strictly increasing class boundaries for ``values``."""
    sorted_values = sorted(float(v) for v in values)
    if not sorted_values:
        return []

    num_classes = max(1, min(int(num_classes), len(sorted_values)))

    if int(size_mode) == MODE_NATURAL_BREAKS:
        breaks = jenks_breaks(sorted_values, num_classes)
    elif int(size_mode) == MODE_QUANTILE:
        breaks = quantile_breaks(sorted_values, num_classes)
    else:
        breaks = equal_interval_breaks(sorted_values, num_classes)

    compact = [float(breaks[0])]
    for value in breaks[1:]:
        fv = float(value)
        if fv > compact[-1]:
            compact.append(fv)
    if len(compact) == 1:
        compact.append(compact[0] + 1.0)
    return compact


def equal_interval_breaks(sorted_values, num_classes):
    """Equal-interval class boundaries."""
    min_val = float(sorted_values[0])
    max_val = float(sorted_values[-1])
    if max_val == min_val:
        return [min_val, max_val]

    step = (max_val - min_val) / float(num_classes)
    breaks = [min_val]
    for i in range(1, num_classes):
        breaks.append(min_val + (step * i))
    breaks.append(max_val)
    return breaks


def quantile_breaks(sorted_values, num_classes):
    """Quantile class boundaries (linear interpolation between ranks)."""
    n = len(sorted_values)
    breaks = [float(sorted_values[0])]
    if n == 1:
        breaks.append(float(sorted_values[0]))
        return breaks

    for i in range(1, num_classes):
        pos = (n - 1) * (float(i) / float(num_classes))
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            q = float(sorted_values[low])
        else:
            weight = pos - low
            q = float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)
        breaks.append(q)

    breaks.append(float(sorted_values[-1]))
    return breaks


def jenks_breaks(sorted_values, num_classes):
    """Natural-breaks (Jenks) class boundaries."""
    n = len(sorted_values)
    if n == 0:
        return []
    if num_classes <= 1:
        return [float(sorted_values[0]), float(sorted_values[-1])]

    lower = [[0] * (num_classes + 1) for _ in range(n + 1)]
    variance = [[float("inf")] * (num_classes + 1) for _ in range(n + 1)]

    for i in range(1, num_classes + 1):
        lower[1][i] = 1
        variance[1][i] = 0.0
        for j in range(2, n + 1):
            variance[j][i] = float("inf")

    for row_end in range(2, n + 1):
        sum_val = 0.0
        sum_sq = 0.0
        w = 0.0
        variance_l = 0.0

        for m in range(1, row_end + 1):
            idx = row_end - m + 1
            val = float(sorted_values[idx - 1])

            w += 1.0
            sum_val += val
            sum_sq += val * val
            variance_l = sum_sq - ((sum_val * sum_val) / w)

            if idx == 1:
                continue

            for j in range(2, num_classes + 1):
                candidate = variance_l + variance[idx - 1][j - 1]
                if variance[row_end][j] >= candidate:
                    lower[row_end][j] = idx
                    variance[row_end][j] = candidate

        lower[row_end][1] = 1
        variance[row_end][1] = variance_l

    breaks = [0.0] * (num_classes + 1)
    breaks[num_classes] = float(sorted_values[-1])
    breaks[0] = float(sorted_values[0])

    k = n
    for j in range(num_classes, 1, -1):
        idx = int(lower[k][j]) - 2
        idx = max(0, idx)
        breaks[j - 1] = float(sorted_values[idx])
        k = int(lower[k][j] - 1)
        if k <= 0:
            break

    return breaks
