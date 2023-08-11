import torch
import glob
from utils.datasets.RawStreamData import RawStreamDS
from utils.functions_SX import get_ideal_basis
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from models.indexer_model import IndexerModule

import lovely_tensors as lt

lt.monkey_patch()

import os

cwd = os.getcwd()

# streams_path = cwd+"/data/lyso_12p4kev_1khz_150mm_run000026"
streams_path = cwd+"/data/performance_test"
mylist = glob.glob(streams_path + '/*.stream', recursive=True)
print("List of stream files to be used", mylist)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print("Using device ", device)


batch_size = 40
spot_sequence_length = 80

# performing params
lattice_size = 50000
angle_resolution = 150
num_top_solutions = 400


# # fast params
# lattice_size = 30000
# angle_resolution = 150
# num_top_solutions = 200


im = IndexerModule(
    lattice_size=lattice_size,
    num_iterations=5,
    error_precision=0.0012,
    filter_precision=0.00075,
    filter_min_num_spots=6
)



for path in mylist:
    spot_sequence_length = spot_sequence_length
    mds = RawStreamDS(path, spot_sequence_length, no_padding=False)
    data_loader = DataLoader(mds, batch_size=batch_size, shuffle=False)

    cell_parameters = mds.instances[0]['initial_cell']
    initial_cell = get_ideal_basis(cell_parameters)

    solution_triples_list = []
    solution_indices_list = []

    wlen = mds.instances[0]['wavelength']
    with torch.no_grad():

        for source, y, grid, indices in tqdm(data_loader):
            source = source.to(device)
            y = y.to(device)
            grid = grid.to(device)

            solution_successes, solution_triples, solution_masks, solution_errors, solution_penalization = im(
                source,
                initial_cell,
                min_num_spots=8,
                angle_resolution=angle_resolution,
                num_top_solutions=num_top_solutions
            )

            for batch, (success_crystals, matrix_crystals) in enumerate(zip(solution_successes, solution_triples)):
                idx = indices[batch].item()
                serial = torch.tensor(mds.instances[idx]['Image serial number'])

                for success, matrix in zip(success_crystals, matrix_crystals):
                    if success:
                        solution_triples_list.append(matrix)
                        solution_indices_list.append(serial)


    print(len(solution_triples_list), " found out of ", len(mds))
