# TORO Indexer for serial crystallography

Serial Crystallography (SX) involves the processing of thousands of diffraction patterns in random orientations. 
To compile a complete data set, these patterns must be indexed, integrated, and merged. 
Herein, we introduce TORO (Torch-Powered Robust Optimization) Indexer, a robust, adaptable, and performant indexing algorithm. 
Developed using the PyTorch framework, TORO Indexer can operate on GPUs, CPUs, TPUs, and other accelerators supported by PyTorch, 
ensuring it adapts to a broad range of computational setups.

## Introduction

The entire model code lies within 500 lines stored in models/indexer_model.py. In there, the class ToroIndexer is all you need to run indexing.

## Installing TORO
Just clone this repository, install the requirements and follow the instructions below to run the indexer.
Click the `Settings` tab where you will find the
git repo URLs - use `git` to clone the project on whichever machine you want.
Two streams of data are provided with this repository to try the indexer, a set with 10000 frames from the Lyso molecule, 
and the dataset we used for our benchmarks called `performance_test`. You can find its description in the accompanying paper.  
Go into the repository and run `pip install -e .` to install the current version of toro. 

If you want to try it without installing it locally, from the Renku platform click on the `Sessions` tab and start a new session.
This will start an interactive environment right in your browser. 
However it might be that no GPU is available, and only limited CPU performance can be tested.




### Trying the indexer
To try the indeer run the command `python stream_file_indexer.py --model=real_time --batch_size=100` , the batch size is recommended to be as high as your GPU alows it.
In our accompanying paper we benchmarked 2 predetermined configurations for the TORO: `precise` and `real_time` with different trade-offs between speed and the number of indexable frames found.
If you want to test performance of the indexer, you can add the flag `--speed_test`, which will disable the storing of the solutions and preloads the dataset into memory to test only the performance of the indexer.

### Visualizing the results
Once you have run the `python stream_file_indexer.py` command (without the `--speed_test` flag), 
you can visualize the results using the notebook `notebooks/results_visualization.ipynb`. 
Simply run all the cells, and reload the last one to visualize random solutions of the indexer.