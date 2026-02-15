HDF5 to Multiscale OME-Zarr Conversion Pipeline

Overview

This repository contains a full pipeline for converting large-scale microscopy datasets into multiresolution OME-Zarr format using block-wise processing and Dask-based parallelism.

The system includes:
• A graphical user interface (GUI) for configuring conversion parameters
• A memory-aware Dask backend for controlled parallel execution
• A block-wise multiscale pyramid generator
• Support for execution on both local machines and HPC compute nodes

⸻

Features
• Block-wise conversion to OME-Zarr
• Multiscale pyramid generation
• Configurable chunk sizes
• Progress reporting
• Visualisation via Napari
• GUI-based execution
• HPC cluster compatibility (tested on DTU HPC)

⸻

Repository Structure

.
├── app.py # Main GUI application entry point
├── requirements.txt # Python dependencies
├── conversionHandling/ # Conversion pipeline
├── conversionHandling/helpers # Algorithms and utilities
└── README.md

⸻

Installation (Local Machine)

1. Clone the Repository

git clone [<repository_url>](https://github.com/tmikkel/HDF5toOME-Zarr.git)
cd HDF5toOME-zarr

2. Create Conda Environment

conda create -n <env_name> python=3.11
conda activate <env_name>

3. Install Dependencies

pip install -r requirements.txt

⸻

Running the Application

Start the GUI:

python app.py

You can configure:
• Chunk sizes
• Memory limits
• Worker count
• Output location

The backend computation will run using Dask.

⸻

Running on DTU HPC Cluster

1. Enable X11 (macOS only)

Open -a XQuartz

Enable:

XQuartz → Settings → Security → Allow connections from network clients

⸻

2. Connect via SSH with X forwarding

ssh -X <username>@login1.gbar.dtu.dk

⸻

3. Install Miniconda (if needed)

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

Restart shell:

source ~/.bashrc

⸻

4. Create Environment

conda create -n <env_name> python=3.11
conda activate <env_name>
pip install -r requirements.txt

⸻

5. Run on Interactive Compute Node (Recommended)

Do not code on the login node.

Request an interactive node via LSF:

bsub -Is -n 4 -R "rusage[mem=8GB]" bash

Then run:

python app.py
