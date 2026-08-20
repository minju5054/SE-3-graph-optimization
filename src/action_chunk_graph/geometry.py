import numpy as np
from scipy.spatial.transform import Rotation


def wrap_angle(angle):
    """Map angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def se2_relative_log(pose_a, pose_b):
    """
    Compute Log(T_a^{-1} T_b) for SE(2).

    pose = [x, y, theta]
    return = [rho_x, rho_y, theta_error]
    """
    xa, ya, tha = pose_a
    xb, yb, thb = pose_b

    c = np.cos(tha)
    s = np.sin(tha)

    dx = xb - xa
    dy = yb - ya

    tx = c * dx + s * dy
    ty = -s * dx + c * dy

    dtheta = float(wrap_angle(thb - tha))

    if abs(dtheta) < 1e-8:
        return np.array([tx, ty, dtheta], dtype=float)

    A = np.sin(dtheta) / dtheta
    B = (1.0 - np.cos(dtheta)) / dtheta

    det = A * A + B * B
    rho_x = (A * tx + B * ty) / det
    rho_y = (-B * tx + A * ty) / det

    return np.array([rho_x, rho_y, dtheta], dtype=float)


def heading_from_xy(x, y):
    """Compute tangent heading for a 2D trajectory."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.unwrap(np.arctan2(dy, dx))


def skew(vector):
    """Return the 3x3 skew-symmetric matrix for a 3-vector."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def so3_left_jacobian(rotation_vector):
    """SO(3) left Jacobian used by the SE(3) exponential map."""
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    theta = np.linalg.norm(rotation_vector)
    phi_hat = skew(rotation_vector)
    phi_hat_squared = phi_hat @ phi_hat

    if theta < 1e-8:
        return np.eye(3) + 0.5 * phi_hat + phi_hat_squared / 6.0

    theta_squared = theta * theta
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta_squared * phi_hat
        + (theta - np.sin(theta)) / (theta_squared * theta) * phi_hat_squared
    )


def so3_left_jacobian_inverse(rotation_vector):
    """Numerically stable inverse of the SO(3) left Jacobian."""
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    theta = np.linalg.norm(rotation_vector)
    phi_hat = skew(rotation_vector)
    phi_hat_squared = phi_hat @ phi_hat

    if theta < 1e-4:
        coefficient = 1.0 / 12.0 + theta * theta / 720.0
    else:
        coefficient = (
            1.0 - 0.5 * theta / np.tan(0.5 * theta)
        ) / (theta * theta)

    return np.eye(3) - 0.5 * phi_hat + coefficient * phi_hat_squared


def se3_exp(twist):
    """Map a twist [rho, phi] in se(3) to a 4x4 transformation."""
    twist = np.asarray(twist, dtype=float)
    if twist.shape != (6,):
        raise ValueError("Expected twist shape [6] with [rho, rotation_vector].")

    rho = twist[:3]
    rotation_vector = twist[3:]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    transform[:3, 3] = so3_left_jacobian(rotation_vector) @ rho
    return transform


def se3_log(transform):
    """Map a 4x4 SE(3) transformation to a twist [rho, phi]."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Expected an SE(3) transform with shape [4, 4].")

    rotation_vector = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    rho = so3_left_jacobian_inverse(rotation_vector) @ transform[:3, 3]
    return np.concatenate([rho, rotation_vector])


def pose6_to_matrix(pose):
    """Convert [x, y, z, rotation-vector] to a 4x4 transformation."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,):
        raise ValueError("Expected pose shape [6] with [x, y, z, rotation_vector].")

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose6(transform):
    """Convert a 4x4 transformation to [x, y, z, rotation-vector]."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Expected an SE(3) transform with shape [4, 4].")
    rotation_vector = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return np.concatenate([transform[:3, 3], rotation_vector])


def se3_relative_log(pose_a, pose_b):
    """Compute Log(T_a^-1 T_b) for poses [x, y, z, rotation-vector]."""
    pose_a = np.asarray(pose_a, dtype=float)
    pose_b = np.asarray(pose_b, dtype=float)
    if pose_a.shape != (6,) or pose_b.shape != (6,):
        raise ValueError("Expected pose shapes [6] with [x, y, z, rotation_vector].")

    rotation_a = Rotation.from_rotvec(pose_a[3:])
    rotation_b = Rotation.from_rotvec(pose_b[3:])
    relative_rotation = rotation_a.inv() * rotation_b
    rotation_vector = relative_rotation.as_rotvec()
    relative_translation = rotation_a.inv().apply(pose_b[:3] - pose_a[:3])
    rho = so3_left_jacobian_inverse(rotation_vector) @ relative_translation
    return np.concatenate([rho, rotation_vector])
