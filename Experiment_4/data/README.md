Use HPC central datasets to keep versions aligned with the paper.

module load common-dataset-loader
common-dataset-loader avail
common-dataset-loader load <dataset_path>
ln -s /home/<username>/hpc-shared-data/<dataset_path> /scratch/<username>/
