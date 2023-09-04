import glob
from utils.datasets.RawStreamData import RawStreamDS
from utils.functions_SX import get_ideal_basis
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.indexer_model import ToroIndexer
import os
from argparse import ArgumentParser
import lovely_tensors as lt
lt.monkey_patch()

precise = "precise"
middle = "middle"
fast = "fast"
insane = "insane"

parser = ArgumentParser(description="ToRO testing")
parser.add_argument("--model", type=str, choices={precise, middle, fast, insane}, help="Choose the desired parameters to get the trade-off between speed and performance.")
parser.add_argument("--batch_size", type=int, default=50, help="Choose the largest batch size that fits in your GPU.")
parser.add_argument("--speed_test", action='store_true', help="Activate when testing speed to remove uneccesary actions, no results will be stored, only the speed will be shown")
args = vars(parser.parse_args())

cwd = os.getcwd()
streams_path = cwd+"/data/lyso_12p4kev_1khz_150mm_run000026"
# streams_path = cwd+"/data/performance_test"
mylist = glob.glob(streams_path + '/*.stream', recursive=True)
print("List of stream files to be used", mylist)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print("Using device ", device)

spot_sequence_length = 80

batch_size = args["batch_size"]

if args["model"] == precise:
    # Max precision params
    lattice_size = 50000
    angle_resolution = 150
    num_top_solutions = 400

elif args["model"] == middle:
    # good precision, good speed
    lattice_size = 25000
    angle_resolution = 150
    num_top_solutions = 200

elif args["model"] == fast:
    # fast params
    lattice_size = 20000
    angle_resolution = 100
    num_top_solutions = 50

elif args["model"] == insane:
    # super fast params
    lattice_size = 10000
    angle_resolution = 100
    num_top_solutions = 25
else:
    raise ValueError("No --model was selected, please run with --model={precise, middle, fast, insane}")


im = ToroIndexer(
    lattice_size=lattice_size,
    num_iterations=5,
    error_precision=0.0012,
    filter_precision=0.00075,
    filter_min_num_spots=6
).to(device)


for path in mylist:
    mds = RawStreamDS(path, spot_sequence_length, no_padding=False)
    data_loader = DataLoader(mds, batch_size=batch_size, shuffle=True)
    if args["speed_test"]:
        # We load the entire dataset into memory already batched before starting the test performance
        print("Loading dataset into memory... You are running a speed_test, results will not be saved.")
        dataset = [(source, indices) for source, indices in tqdm(data_loader)]
    else:
        dataset = data_loader

    cell_parameters = mds.instances[0]['initial_cell']
    initial_cell = get_ideal_basis(cell_parameters)

    solution_triples_list = []
    solution_indices_list = []

    wlen = mds.instances[0]['wavelength']
    with torch.no_grad():
        print("Indexing...")
        for source, indices in tqdm(dataset):
            source = source.to(device)
            solution_successes, solution_triples, solution_masks, solution_errors, solution_penalization = im(
                source,
                initial_cell,
                min_num_spots=8,
                angle_resolution=angle_resolution,
                num_top_solutions=num_top_solutions
            )
            if not args["speed_test"]:
                for batch, (success_crystals, matrix_crystals) in enumerate(zip(solution_successes, solution_triples)):
                    idx = indices[batch].item()
                    serial = torch.tensor(mds.instances[idx]['Image serial number'])

                    for success, matrix in zip(success_crystals, matrix_crystals):
                        if success:
                            solution_triples_list.append(matrix)
                            solution_indices_list.append(serial)

    if not args["speed_test"]:
        found_indices = torch.stack(solution_indices_list)
        found_triples = torch.stack(solution_triples_list)
        np.save("solution_indices", found_indices.to('cpu').numpy())
        np.save("solution_matrices", found_triples.to('cpu').numpy())
        print(len(solution_triples_list), " found out of ", len(mds))
        print("Solutions have been saved, you can view them using the provided notebook: results_visualization.ipynb")
