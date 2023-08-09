import torch
import numpy as np
from collections import defaultdict
from utils.functions_SX import map_3D
# from crystfelparser.crystfelparser import stream_to_dictionary

def stream_to_dictionary(streamfile):
    """
    Function for parsing a indexamajig output stream

    Args:
      h5file: stream file to parse

    Returns:
      A dictionary
    """
    series = defaultdict(dict)
    series = dict()

    def loop_over_next_N_lines(file, n_lines):

        for cnt_tmp in range(n_lines):
            line = file.readline()

        return line

    with open(streamfile, "r") as text_file:
        # for ln,line in enumerate(text_file):
        ln = -1
        while True:
            ln += 1
            line = text_file.readline()
            # if any(x in ["Begin","chunk"] for x in line.split()):
            if "Begin chunk" in line:
                # create a temporary dictionary to store the output for a frame
                # tmpframe = defaultdict(int)
                tmpframe = dict()

                # loop over the next 3 lines to get the index of the image
                # line 2 and 3 are where it is stored the image number
                line = loop_over_next_N_lines(text_file, 3)
                ln += 3
                # save the image index and save it as zero-based
                im_num = np.int(line.split()[-1]) - 1
                tmpframe["Image serial number"] = im_num

                # loop over the next 2 lines to see if the indexer worked
                line = loop_over_next_N_lines(text_file, 2)
                ln += 2
                # save who indexed the image
                indexer_tmp = line.split()[-1]
                # if indexed, there is an additional line here
                tmpframe["indexed_by"] = indexer_tmp

                ##### Get the STRONG REFLEXTIONS from the spotfinder #####
                keyw = ""
                while keyw != "num_peaks":
                    # loop over the next 5/6 lines to get the number of reflctions
                    line = loop_over_next_N_lines(text_file, 1)
                    ln += 1
                    try:
                        keyw = line.split()[0]
                    except:
                        keyw = ""

                # get the number of peaks
                num_peaks = np.int(line.split()[-1])
                tmpframe["num_peaks"] = num_peaks

                # get the resolution
                line = text_file.readline()
                ln += 1
                tmpframe["peak_resolution [A]"] = np.float(line.split()[-2])
                tmpframe["peak_resolution [nm^-1]"] = np.float(line.split()[2])

                if num_peaks > 0:
                    # skip the first 2 lines
                    for tmpc in range(2):
                        text_file.readline()
                        ln += 1

                    # get the spots
                    # fs/px, ss/px, (1/d)/nm^-1, Intensity
                    # with
                    # dim1 = ss, dim2 = fs
                    tmpframe["peaks"] = np.asarray(
                        [text_file.readline().split()[:4] for tmpc in range(num_peaks)]
                    ).astype(np.float)

                ##### Get the PREDICTIONS after indexing #####

                if tmpframe["indexed_by"] != "none":
                    # skip the first 2 header lines
                    for tmpc in range(2):
                        text_file.readline()
                        ln += 1
                    # Get the unit cell -- as cell lengths and angles
                    line = text_file.readline().split()
                    tmpframe["Cell parameters"] = np.hstack(
                        [line[2:5], line[6:9]]
                    ).astype(np.float)

                    # Get the reciprocal unit cell as a 3x3 matrix
                    reciprocal_cell = []
                    for tmpc in range(3):
                        reciprocal_cell.append(text_file.readline().split()[2:5])
                        ln += 1
                    tmpframe["reciprocal_cell_matrix"] = np.asarray(
                        reciprocal_cell
                    ).astype(np.float)

                    # Save the lattice type
                    tmpframe["lattice_type"] = text_file.readline().split()[-1]
                    ln += 1

                    # loop over the next 5 lines to get the diffraction resolution
                    line = loop_over_next_N_lines(text_file, 5).split()
                    ln += 5

                    if line[0] == "predict_refine/det_shift":
                        tmpframe["det_shift_x"] = line[3]
                        tmpframe["det_shift_y"] = line[6]
                        line = loop_over_next_N_lines(text_file, 1).split()
                        ln += 1

                    tmpframe["diffraction_resolution_limit [nm^-1]"] = np.float(line[2])
                    tmpframe["diffraction_resolution_limit [A]"] = np.float(line[5])

                    # get the number of predicted reflections
                    num_reflections = np.int(text_file.readline().split()[-1])
                    tmpframe["num_predicted_reflections"] = num_reflections

                    # skip a few lines
                    line = loop_over_next_N_lines(text_file, 4)
                    ln += 4
                    # get the predicted reflections
                    if num_reflections > 0:
                        reflections_pos = []
                        for tmpc in range(num_reflections):
                            # read as:
                            # h    k    l          I   sigma(I)       peak background  fs/px  ss/px
                            line = np.asarray(text_file.readline().split()[:9])
                            # append only:   fs/px  ss/px  I sigma(I)
                            reflections_pos.append(line[[7, 8, 3, 4, 0, 1, 2]])
                            ln += 1
                        tmpframe["predicted_reflections"] = np.asarray(
                            reflections_pos
                        ).astype(np.float)
                    # continue reading
                    line = text_file.readline()
                    ln += 1

                # Add the frame to the series, using the frame index as key
                series[im_num] = tmpframe

            # condition to exit the while true reading cycle
            if "" == line:
                # print("file finished")
                break

    # return the series
    return series

def get_experiment_info(streamfile):
    '''
    get same info about the experiment from the stream file

    Input:
       streamfile:    path to the stream file

    Outputs:
       abs(posx):     beam center's X coordinate
       abs(posy):     beam center's Y coordinate
       clen:          detector distance
       photon_energy: energy of a photon in Angstroms
       cell:          initial guess for the unit cell

    '''
    import subprocess
    # get the x
    proc = subprocess.Popen("grep corner_x {}".format(streamfile), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode('UTF-8')
    posx = float(out.split()[2])
    # get the y
    proc = subprocess.Popen("grep corner_y {}".format(streamfile), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode('UTF-8')
    # get the detector distance
    posy = float(out.split()[2])
    proc = subprocess.Popen("grep clen {}".format(streamfile), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode('UTF-8')
    clen = float(out.split()[2]) * 1000
    # get the photon energy
    proc = subprocess.Popen("grep 'photon_energy' {}".format(streamfile), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode('UTF-8')
    # convert to angstrom
    photon_energy = 12398.42 / np.float(max([int(i) for i in set(out.split()) if i.isnumeric()]))
    proc = subprocess.Popen("grep -A11 'Begin unit cell'  {}".format(streamfile), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode('UTF-8')
    cellstr = [line for line in out.split("\n") if len(line) > 0 if len(line) > 1 if
               line.split()[0] in {"a", "b", "c", "al", "be", "ga"}]
    cell = np.array([np.float(line.split()[2]) for line in cellstr])

    return abs(posx), abs(posy), clen, photon_energy, cell


class RawStreamDS(torch.utils.data.Dataset):

    def __init__(self, file, sequence_length, no_padding=True, spot_sorting=False, nx=2067, ny=2163, pixel_size=0.075):
        super(RawStreamDS, self).__init__()
        self.spot_sorting = spot_sorting
        self.no_padding = no_padding
        #pixel size in mm
        self.pixel_size = pixel_size
        # size of the image in pixels
        self.nx = nx
        self.ny = ny

        print("Creating DS from ", file)
        xbeam, ybeam, detector_distance, wavelength, initial_cell = get_experiment_info(file)
        self.instances = stream_to_dictionary(file)

        for key in self.instances:
            self.instances[key]["initial_cell"] = initial_cell
            self.instances[key]["wavelength"] = wavelength
            self.instances[key]["nx"] = self.nx  # pixels
            self.instances[key]["ny"] = self.ny  # pixels
            self.instances[key]["qx"] = self.pixel_size # mm
            self.instances[key]["qy"] = self.pixel_size  # mm
            self.instances[key]["orgx"] = xbeam
            self.instances[key]["orgy"] = ybeam
            self.instances[key]["det_dist"] = detector_distance
            self.instances[key]["det_x"] = np.array([1., 0., 0.])
            self.instances[key]["det_y"] = np.array([0., 1., 0.])

        self.sequence_length = sequence_length


    def project_onto_sphere(self, data, param):
        sp3d = map_3D(
            data,
            oscillation_range=0.0,
            wavelength=param["wavelength"],
            nx=param["nx"],
            ny=param["ny"],
            qx=param["qx"],
            qy=param["qy"],
            orgx=param["orgx"],
            orgy=param["orgy"],
            det_dist=param["det_dist"],
            det_x=param["det_x"],
            det_y=param["det_y"],
        )
        return torch.FloatTensor(sp3d)

    def __getitem__(self, idx):
        with torch.no_grad():
            frame = self.instances[idx]
            # if it has no peaks then we need to return empty lists
            if not 'peaks' in frame:
                lattice_points = torch.ones(1, 8)
            else:
                lattice_points = self.project_onto_sphere(frame['peaks'], frame)
            num_lattice = len(lattice_points)
            num_random = max(0, self.sequence_length - num_lattice)
            if self.no_padding:
                num_random = 0

            rp_sp3D = torch.zeros(num_random, lattice_points.shape[1])

            joined = [lattice_points, rp_sp3D]
            x = torch.cat(joined, 0)
            if 'labels' in frame.keys():
                grid = torch.FloatTensor(frame['labels'])
            else:
                grid = torch.FloatTensor(x[:, :3] * 0)

            y = (torch.sum(torch.abs(grid), dim=-1) != 0).long()

            if num_random > 0:
                grid_random = rp_sp3D[:, :3] * 0
                y_random = rp_sp3D[:, 0] * 0
                y = torch.cat([y, y_random], 0)
                grid = torch.cat([grid, grid_random], 0)

            if self.spot_sorting:
                sorting = x.norm(dim=-1).sort(descending=False).indices[:self.sequence_length]
                x = x[sorting]
                y = y[sorting]
                grid = grid[sorting]
            else:
                x = x[:self.sequence_length]
                y = y[:self.sequence_length]
                grid = grid[:self.sequence_length]
            assert len(x) == len(y)

            x = x[:, :3]

            return x, y, grid, idx

    def get_idx_from_serial(self, serial_number):
        return serial_number

    def __len__(self):
        return len(self.instances)
