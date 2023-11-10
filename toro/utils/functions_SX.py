import numpy as np
import torch
import math
import matplotlib.pyplot as plt

def print_solution(best_triple, idx, mds, initial_cell):
    with torch.no_grad():
        wlen = mds.instances[idx]["wavelength"]
        factor = 0.0005
        predicted_spots = predictBraggsReflections(
            torch.inverse(best_triple.T).to('cpu'),
            vwlen=wlen,
            maxhkl=max(initial_cell.norm(dim=-1)) // 2,
            detector_distance_m=mds.instances[idx]["det_dist"] / 1000,
            detectorCenter=[mds.instances[idx]["orgx"], mds.instances[idx]["orgy"]],
            pixelLength_m=75e-6,
            nx=mds.instances[idx]["nx"], ny=mds.instances[idx]["ny"],
            eps=factor, fromXgandalf=True
        )

        if 'reciprocal_cell_matrix' in mds.instances[idx]:
            predicted_xgandalf = predictBraggsReflections(
                torch.FloatTensor(mds.instances[idx]['reciprocal_cell_matrix']) / 10,
                vwlen=wlen,
                maxhkl=max(initial_cell.norm(dim=-1)) // 2,
                detector_distance_m=mds.instances[idx]["det_dist"] / 1000,
                detectorCenter=[mds.instances[idx]["orgx"], mds.instances[idx]["orgy"]],
                pixelLength_m=75e-6,
                nx=mds.instances[idx]["nx"], ny=mds.instances[idx]["ny"],
                eps=factor,

            )

    labeled_peaks = mds.instances[idx]['peaks'][:, :2]
    plt.figure(figsize=(14, 14))
    plt.title(
        "Indexed diffraction pattern. Spots are depicted in blue, the predicted reflections from the indexing are shown in magenta. \n In green is the predicted reflections found by Xgandalf in case it is in the stream file.")
    plt.scatter(
        mds.instances[idx]['orgx'] - 1 * (labeled_peaks[:, 0] - mds.instances[idx]['orgx']),
        labeled_peaks[:, 1], s=10, facecolors='b', edgecolors='none')

    if 'predicted_reflections' in mds.instances[idx]:
        predicted = mds.instances[idx]['predicted_reflections'][:, :2]
        plt.scatter(predicted_xgandalf[:, 0], predicted_xgandalf[:, 1], s=140, facecolors='none', edgecolors='g')

    plt.scatter(predicted_spots[:, 0], predicted_spots[:, 1], s=60, facecolors='none', edgecolors='m')

    plt.show()


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

def predictBraggsReflections(reciprocal_basis_vectors,
                             vwlen=0.9998725806451613,
                             maxhkl=30,
                             detector_distance_m=0.200,
                             detectorCenter=[1122.215602, 1170.680571],
                             pixelLength_m=75e-6,
                             nx=2068, ny=2164,
                             eps=4e-4,
                             device='cpu',
                             fromXgandalf: bool = False):
    """ Given a rotation matrix in the reciprocal space, predict the ideal Bragg's reflections
    Args:
        reciprocal_basis_vectors (torch.tensor): reciprocal basis vectors (a*, b*, c*)
        vwlen (float, optional): beam's wavelenght. Defaults to 0.9998725806451613.
        maxhkl (int, optional): max integer for h, k, l numbers. Defaults to 30.
        detector_distance_m (float, optional): Detector distance in m. Defaults to 0.200.
        detectorCenter (list, optional): Beam's center in pixels. Defaults to [1122.215602,1170.680571].
        pixelLength_m (float, optional): Pixels size in m. Defaults to 75e-6.
        nx (int, optional): Detector size (x direction). Defaults to 2068.
        ny (int, optional): Detector size (y direction). Defaults to 2164.
        eps (float, optional): Tolerance (correlates with the non-monocromaticity). Defaults to 3e-4.
        device (str, optional): Torch device. Defaults to 'cpu'.
    Returns:
        _type_: _description_
    """
    #  for now this will stay hard-coded
    beam_direction = [0, 0, 1 / vwlen]
    # generate the millers grid
    rh = torch.arange(-maxhkl, maxhkl + 1, dtype=torch.int16, device=device)
    rk = torch.arange(-maxhkl, maxhkl + 1, dtype=torch.int16, device=device)
    rl = torch.arange(-maxhkl, maxhkl + 1, dtype=torch.int16, device=device)
    millers = torch.stack([
        rh.repeat_interleave(len(rk) * len(rl)),
        rk.repeat(len(rh)).repeat_interleave(len(rl)),
        rl.repeat(len(rh) * len(rk)),
    ]).T
    # generate the reciprocal lattice vectors
    reciprocalPeaks = millers.float() @ reciprocal_basis_vectors
    # center on the Ewald sphere
    reciprocalPeaks += torch.tensor(beam_direction)
    # threshold
    cond = torch.abs(torch.norm(reciprocalPeaks, dim=1) - (1 / vwlen)) < eps
    reciprocalPeaks = reciprocalPeaks[cond]
    reciprocalPeaks += torch.tensor(beam_direction)
    # reflect the x-axis
    # rotation matrix
    if fromXgandalf:
        rt = torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]], dtype=torch.double)
        reciprocalPeaks = reciprocalPeaks @ rt.float()
    return project_to_detector(vwlen, reciprocalPeaks, detector_distance_m, detectorCenter, pixelLength_m, nx, ny)


def project_to_detector(
        vwlen,
        reciprocalPeaks,
        detector_distance_m,
        detectorCenter,
        pixelLength_m,
        nx,
        ny,
        prune_in_frame=True
):
    """ Given points in reciprocal space on the Ewald sphere, it projects them down to the detector
        Args:
            vwlen (float, optional): beam's wavelenght. Defaults to 0.9998725806451613.
            reciprocalPeaks (torch.tensor): points in reciprocal space
            detector_distance_m (float, optional): Detector distance in m. Defaults to 0.200.
            detectorCenter (list, optional): Beam's center in pixels. Defaults to [1122.215602,1170.680571].
            pixelLength_m (float, optional): Pixels size in m. Defaults to 75e-6.
            nx (int, optional): Detector size (x direction). Defaults to 2068.
            ny (int, optional): Detector size (y direction). Defaults to 2164.
        Returns:
            _type_: _description_
        """
    reciprocalPeaks = reciprocalPeaks.clone()
    beam_direction = [0, 0, 1 / vwlen]
    reciprocalPeaks *= torch.tensor([-1, 1, 1])
    detectorCenter = [detectorCenter[1], detectorCenter[0]]
    # recenter the points
    # reciprocalPeaks += 2 * torch.tensor([beam_direction])
    # flip
    projectedPeaks = reciprocalPeaks[:, [1, 0]] / (reciprocalPeaks[:, -1:] - (1 / vwlen)) * detector_distance_m
    # center and flip x and y
    projectedPeaks = (projectedPeaks / pixelLength_m + torch.tensor(detectorCenter, device=reciprocalPeaks.device))[:,
                     [1, 0]]
    # limit the points inside the detector panel
    mask = (projectedPeaks[:, 0] > 0) & (projectedPeaks[:, 0] < nx) & (projectedPeaks[:, 1] > 0) & (
            projectedPeaks[:, 1] < ny)
    return projectedPeaks[mask]