import numpy as np

def behavior_acc(behavior):
    return np.mean(behavior["ACC"])

def entropy(behavior):
    breakpoint()

def matching_ranges(good_matching_steps):
    ranges = []
    step_b = good_matching_steps[0]
    step_c = good_matching_steps[0]
    for step in good_matching_steps[1:]:
        if step == step_c + 1:
            step_c = step
        else:
            ranges.append((step_b, step_c))
            step_b = step
            step_c = step
    if step_c == good_matching_steps[-1]:
        ranges.append((step_b, step_c))
    return ranges

def cohen_d(d1, d2, axis = None):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1, axis = axis), np.var(d2, ddof=1, axis = axis)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1, axis = axis), np.mean(d2, axis = axis)
    return (u1 - u2) / s