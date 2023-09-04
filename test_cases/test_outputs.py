import torch
import glob
from utils.datasets.RawStreamData import RawStreamDS
from utils.functions_SX import get_ideal_basis
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset
from models.indexer_model import ToroIndexer

import lovely_tensors as lt

lt.monkey_patch()

import os

cwd = os.getcwd()

streams_path = cwd+"/../data/lyso_12p4kev_1khz_150mm_run000026"
# streams_path = cwd+"/../data/performance_test"
mylist = glob.glob(streams_path + '/*.stream', recursive=True)
print("List of stream files to be used", mylist)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print("Using device ", device)



spot_sequence_length = 80

# performing params
batch_size = 5
lattice_size = 50000
angle_resolution = 150
num_top_solutions = 400


# # fast params
# batch_size = 100
# lattice_size = 25000
# angle_resolution = 150
# num_top_solutions = 200

# # super fast params
# batch_size = 200
# lattice_size = 20000
# angle_resolution = 100
# num_top_solutions = 50


im = ToroIndexer(
    lattice_size=lattice_size,
    num_iterations=5,
    error_precision=0.0012,
    filter_precision=0.00075,
    filter_min_num_spots=6
).to(device)
im.debugging = True

im_cpu = ToroIndexer(
    lattice_size=lattice_size,
    num_iterations=5,
    error_precision=0.0012,
    filter_precision=0.00075,
    filter_min_num_spots=6
).to('cpu')
im_cpu.debugging = True



for path in mylist:
    mds = RawStreamDS(path, spot_sequence_length, no_padding=False)
    data_loader = DataLoader(mds, batch_size=batch_size, shuffle=False)

    cell_parameters = mds.instances[0]['initial_cell']
    initial_cell = get_ideal_basis(cell_parameters)

    solution_triples_list = []
    solution_indices_list = []

    wlen = mds.instances[0]['wavelength']
    with torch.no_grad():


        for source, indices in tqdm(data_loader):
            source = source.to(dtype=torch.float32)
            solution_successes_cpu, solution_triples_cpu, solution_masks_cpu, solution_errors_cpu, solution_penalization_cpu = im_cpu(
                source[0].unsqueeze(0),
                initial_cell,
                min_num_spots=8,
                angle_resolution=angle_resolution,
                num_top_solutions=num_top_solutions
            )

            source = source.to(device, dtype=torch.float32)
            solution_successes, solution_triples, solution_masks, solution_errors, solution_penalization = im(
                source,
                initial_cell,
                min_num_spots=8,
                angle_resolution=angle_resolution,
                num_top_solutions=num_top_solutions
            )

            assert(torch.allclose(im.unit_candidates[0],  im_cpu.unit_candidates[0], atol=1e-5))
            assert(torch.allclose(im.projections[:,0], im_cpu.projections[:, 0], atol=1e-5))
            assert(torch.allclose(im.all_candidates_prior[:, 0],  im_cpu.all_candidates_prior[:, 0], atol=1e-5))
            assert(torch.allclose(im.all_candidates_post[:, 0][~im.all_candidates_post[:, 0].isnan()],  im_cpu.all_candidates_post[:, 0][~im_cpu.all_candidates_post[:, 0].isnan()], atol=1e-5))
            assert(torch.allclose(im.candidates[:, 0][~im.candidates[:, 0].isnan()],  im_cpu.candidates[:, 0][~im_cpu.candidates[:, 0].isnan()], atol=1e-5))
            assert(torch.allclose(im.raw_bases[0][~im.raw_bases[0].isnan()], im_cpu.raw_bases[0][~im_cpu.raw_bases[0].isnan()], atol=1e-5))
            assert(torch.allclose(im.filtered_bases[0][~im.filtered_bases[0].isnan()], im_cpu.filtered_bases[0][~im_cpu.filtered_bases[0].isnan()], atol=1e-5))
            assert (torch.all(im.is_inlier[0] == im_cpu.is_inlier[0]))


            mask = (~im.top_bases[0].isnan()) & (~im_cpu.top_bases[0].isnan())
            assert(torch.allclose(im.top_bases[0][mask], im_cpu.top_bases[0][mask], atol=1e-5))
            mask = ~im.penalization[0].isnan()
            assert (torch.allclose(im.penalization[0][mask], im_cpu.penalization[0][mask], atol=1e-5))




