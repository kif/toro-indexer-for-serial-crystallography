import numpy as np
import torch
import math

def rodrigues(h, phi, rot_ax):
    import numpy as np

    cp = np.cos(phi)
    sp = np.sin(phi)
    omcp = 1. - cp

    rot_h = np.zeros(3)

    rot_h[0] = (cp + rot_ax[0] ** 2 * omcp) * h[0] + \
               (-rot_ax[2] * sp + rot_ax[0] * rot_ax[1] * omcp) * h[1] + \
               (rot_ax[1] * sp + rot_ax[0] * rot_ax[2] * omcp) * h[2]
    rot_h[1] = (rot_ax[2] * sp + rot_ax[0] * rot_ax[1] * omcp) * h[0] + \
               (cp + rot_ax[1] ** 2 * omcp) * h[1] + \
               (-rot_ax[0] * sp + rot_ax[1] * rot_ax[2] * omcp) * h[2]
    rot_h[2] = (-rot_ax[1] * sp + rot_ax[0] * rot_ax[2] * omcp) * h[0] + \
               (rot_ax[1] * sp + rot_ax[1] * rot_ax[2] * omcp) * h[1] + \
               (cp + rot_ax[2] ** 2 * omcp) * h[2]

    return rot_h


def map_3D(spotslist,
           wavelength=0.999857,
           # size of the panel
           nx=4148,
           ny=4362,
           # size of the pixels
           qx=0.075000,
           qy=0.075000,
           orgx=2120.750488,
           orgy=2146.885498,
           det_dist=300.0,
           det_x=np.asarray([1.0, 0.0, 0.0]),
           det_y=np.asarray([0.0, 1.0, 0.0]),
           resolmax=0.1,
           resolmin=999,
           starting_angle=0,
           oscillation_range=0.0,
           rot_ax=np.asarray([1.000000, 0, 0])
           ):
    import numpy as np

    incident_beam = np.asarray([0., 0., 1./wavelength])

    det_z = np.zeros(3)
    # comput z in case x,y are not perpendicular to the beam
    det_z[0] = det_x[1] * det_y[2] - det_x[2] * det_y[1]  # calculate detector normal -
    det_z[1] = det_x[2] * det_y[0] - det_x[0] * det_y[2]  # XDS.INP does not have
    det_z[2] = det_x[0] * det_y[1] - det_x[1] * det_y[0]  # this item.
    det_z = det_z / np.sqrt(np.dot(det_z, det_z))  # normalize (usually not req'd)

    spots = []

    for line in spotslist:
        (ih, ik, il) = (0., 0., 0.)
        is_lattice = 0.
        if len(line) == 4:
            (x, y, phi, intensity) = line
        elif len(line) == 8: #stream data
            (x, y, phi, intensity, ih, ik, il, is_lattice) = line
        else:
            (x, y, phi, intensity, ih, ik, il) = line

        # convert detector coordinates to local coordinate system
        r = np.asarray([
            (x - orgx) * qx * det_x[0] + (y - orgy) * qy * det_y[0] + det_dist * det_z[0],
            (x - orgx) * qx * det_x[1] + (y - orgy) * qy * det_y[1] + det_dist * det_z[1],
            (x - orgx) * qx * det_x[2] + (y - orgy) * qy * det_y[2] + det_dist * det_z[2],
        ])

        # normalize scattered vector to obtain S1
        r = r / (wavelength * np.sqrt(np.dot(r, r)))
        # r = r / (np.sqrt(np.dot(r, r)))

        # obtain reciprocal space vector S = S1-S0
        r = r - incident_beam

        if (np.sqrt(np.dot(r, r)) > 1. / resolmax):
            continue  # outer resolution limit
        if (np.sqrt(np.dot(r, r)) < 1. / resolmin):
            continue  # inner resolution limit

        # rotate
        # NB: the term "-180." (found by trial&error) seems to make it match dials.rs_mapper
        phi = (starting_angle + oscillation_range * phi - 180.) / 180. * np.pi

        rot_r = rodrigues(r, phi, rot_ax)

        # rot_r=100.*rot_r + 100./resolmax  # ! transform to match dials.rs_mapper

        spots.append(np.hstack([rot_r, [intensity], [ih, ik, il], is_lattice]))

    return np.asarray(spots)


def get_ideal_basis(cell_parameters):
    a = cell_parameters[0]
    b = cell_parameters[1]
    c = cell_parameters[2]
    alpha = cell_parameters[3] * math.pi / 180
    beta = cell_parameters[4] * math.pi / 180
    gamma = cell_parameters[5] * math.pi / 180

    v_a = torch.tensor([a, 0, 0])
    v_b = torch.tensor([b * math.cos(gamma), b * math.sin(gamma), 0])

    tmp = math.cos(alpha) * math.cos(alpha) + math.cos(beta) * math.cos(beta)
    tmp += math.cos(gamma) * math.cos(gamma) - 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    V = a * b * c * math.sqrt(1.0 - tmp)

    cosalphastar = math.cos(beta) * math.cos(gamma) - math.cos(alpha)
    cosalphastar /= math.sin(beta) * math.sin(gamma)

    cstar = (a * b * math.sin(gamma)) / V

    # /* c in terms of x, y and z */
    v_c = torch.tensor([
        c * math.cos(beta),
        -c * math.sin(beta) * cosalphastar,
        1.0 / cstar
    ])
    basis = torch.stack([v_a, v_b, v_c], 0)
    return basis.float()