import os
import time
import glob
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm
from toro.utils.datasets.RawStreamData import RawStreamDS
from toro.utils.functions_SX import get_ideal_basis
from torch.utils.data import DataLoader
from toro.models.indexer_model import ToroIndexer
import lovely_tensors as lt

lt.monkey_patch()

precise = "precise"
fast = "real_time"

def main():
    parser = ArgumentParser(description="ToRO testing")
    parser.add_argument("--model", type=str, default=precise, choices={precise, fast},
                        help="Choose the desired parameters to get the trade-off between speed and performance.")
    parser.add_argument("--batch_size", type=int, default=50, help="Choose the largest batch size that fits in your GPU.")
    parser.add_argument("--speed_test", action='store_true',
                        help="Activate when testing speed to remove uneccesary actions, no results will be stored, only the speed will be shown")
    parser.add_argument("--cpu", action='store_true', help="Run in CPU instead of GPU")
    args = vars(parser.parse_args())
    
    cwd = os.getcwd()
    # streams_path = cwd + "/data/lyso"
    streams_path = cwd+"/data/performance_test"
    mylist = glob.glob(streams_path + '/*.stream', recursive=True)
    print("List of stream files to be used", mylist)
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    if args['cpu']:
        device = torch.device('cpu')
    print("Using device ", device)
    torch.set_num_threads(10)
    
    spot_sequence_length = 80
    
    batch_size = args["batch_size"]
    
    if args["model"] == precise:
        # Max precision params
        lattice_size = 50000
        angle_resolution = 150
        num_top_solutions = 400
    elif args["model"] == fast:
        # fast params
        lattice_size = 10000
        angle_resolution = 100
        num_top_solutions = 25
    else:
        raise ValueError(f"No --model was selected, please run with --model=({precise}, {fast})")
    
    im = ToroIndexer(
        lattice_size=lattice_size,
        num_iterations=5,
        error_precision=0.0012,
        filter_precision=0.00075,
        filter_min_num_spots=6
    ).to(device)
    
    print("batch_size ", batch_size)
    for path in mylist:
        mds = RawStreamDS(path, spot_sequence_length, no_padding=False)
    
        if args["speed_test"]:
            # If the dataset is small, we go over the dataset a few times to have more relieble metrics
            expanded_mds = torch.utils.data.ConcatDataset((10000 // len(mds)) * [mds])
            print("Size of expanded ds ", len(expanded_mds))
            data_loader = DataLoader(expanded_mds, batch_size=batch_size, shuffle=True)
            # We load the entire dataset into memory already batched before starting the test performance
            print("Loading dataset into memory... You are running a speed_test, results will not be saved.")
            dataset = [(source, indices) for source, indices in tqdm(data_loader)]
        else:
            data_loader = DataLoader(mds, batch_size=batch_size, shuffle=True)
            dataset = data_loader
    
        key0 = list(mds.instances.keys())[0]
        cell_parameters = mds.instances[key0]['initial_cell']
        initial_cell = get_ideal_basis(cell_parameters)
    
        solution_triples_list = []
        solution_indices_list = []
    
        wlen = mds.instances[key0]['wavelength']
        with torch.no_grad():
            if args["speed_test"]:
                # Initialize CUDA events
                if not args['cpu']:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                # Warm-up
                params = [torch.tensor(8), torch.tensor(angle_resolution), torch.tensor(num_top_solutions)]
                im(
                    torch.randn(5, 80, 3, device=device),
                    initial_cell,
                    params
                )
                frames_per_second = []
    
            print("Indexing...")
            for source, indices in tqdm(dataset):
                source = source.to(device)
                if args["speed_test"]:
                    if not args['cpu']:
                        start_event.record()
                    else:
                        start_time = time.perf_counter()
                params = [torch.tensor(8), torch.tensor(angle_resolution), torch.tensor(num_top_solutions)]
                solution_successes, solution_triples, solution_masks, solution_errors, solution_penalization = im(
                    source,
                    initial_cell,
                    params
                )
    
                if args["speed_test"]:
                    if not args['cpu']:
                        end_event.record()
                        # Wait for the events to be recorded
                        torch.cuda.synchronize()
                        elapsed_time = start_event.elapsed_time(end_event)
                    else:
                        end_time = time.perf_counter()
                        elapsed_time = (end_time - start_time) * 1000
    
                    frames_per_second.append(args['batch_size'] * 1000 / elapsed_time)
    
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
            if not args['cpu']:
                name = f"GPU_{torch.cuda.get_device_name(device)}_model_{args['model']}_batch_size_{args['batch_size']}"
            else:
                name = f"CPU_model_{args['model']}_batch_size_{args['batch_size']}"
            np.save("solution_indices", found_indices.to('cpu').numpy())
            np.save("solution_matrices", found_triples.to('cpu').numpy())
            print(len(solution_triples_list), " found out of ", len(mds))
            print("Solutions have been saved, you can visualize them using the provided notebook: results_visualization.ipynb")
        else:
            # Compute average and standard deviation of the time taken
            avg_fs = sum(frames_per_second) / len(frames_per_second)
            std_dev = (sum((x - avg_fs) ** 2 for x in frames_per_second) / len(frames_per_second)) ** 0.5
            if not args['cpu']:
                line = f"GPU:{torch.cuda.get_device_name(device)}, model:{args['model']}, batch_size:{args['batch_size']}. Average number of frames per second: {avg_fs:.2f} f/s, Standard Deviation: {std_dev:.2f} f/s"
            else:
                line = f"CPU, model:{args['model']}, batch_size:{args['batch_size']}. Average number of frames per second: {avg_fs:.2f} f/s, Standard Deviation: {std_dev:.2f} f/s"
            print(line)
            with open("speed_statistics.txt", 'a+') as f:
                f.write(line + '\n')