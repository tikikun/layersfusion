import torch
import numpy as np

def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    epsilon = 1e-10

    # Convert tensors to a common format, float32
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    # Convert tensors to numpy arrays
    c = False
    if not isinstance(v0, np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles    
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)

    if norm_v0 > epsilon:
        v0 = v0 / norm_v0
    else:
        print(f"Warning: Norm of v0 is very small ({norm_v0}). Skipping normalization.")

    if norm_v1 > epsilon:
        v1 = v1 / norm_v1
    else:
        print(f"Warning: Norm of v1 is very small ({norm_v1}). Skipping normalization.")

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    del v0_copy, v1_copy
    del v1

    if c:
        res = torch.from_numpy(v2)
    else:
        res = v2
    return res

