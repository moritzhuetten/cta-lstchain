import numpy as np


def log_gaussian2d(size, x, y, x_cm, y_cm, width, length, psi):

    scale_w = 1. / (2. * width ** 2)
    scale_l = 1. / (2. * length ** 2)
    a = np.cos(psi) ** 2 * scale_l + np.sin(psi) ** 2 * scale_w
    b = np.sin(2 * psi) * (scale_w - scale_l) / 2.
    c = np.cos(psi) ** 2 * scale_w + np.sin(psi) ** 2 * scale_l

    norm = 1. / (2 * np.pi * width * length)

    log_pdf = - (a * (x - x_cm) ** 2 - 2 * b * (x - x_cm) * (y - y_cm) + c * (
                y - y_cm) ** 2)

    log_pdf += np.log(norm) + np.log(size)

    return log_pdf