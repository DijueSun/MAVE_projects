#!/usr/bin/env python3

import os
import sys

# Define paths
paths = {
    "design": "cd /lustre/scratch127/mave/sge_analysis/targeton_design && bash",
    "raw": "cd /lustre/scratch127/mave/sge_analysis/analysis && bash",
    "plasmid": "cd /warehouse/mave/sge_production_02 && bash"
}

host = "ds39@farm5-head2"

# Main function
def run_ssh(path_key):
    if path_key not in paths:
        print("Usage: python farmgo.py [design|raw|plasmid]")
        return
    os.system(f'ssh {host} "{paths[path_key]}"')

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python farmgo.py [design|raw|plasmid]")
    else:
        run_ssh(sys.argv[1])
# open terminal, go to the PycharmProjects/MAVE_Project/ and python farmgo.py design
# open terminal, go to the PycharmProjects/MAVE_Project/ and python farmgo.py raw
# open terminal, go to the PycharmProjects/MAVE_Project/ and python farmgo.py plasmids

# Downloading data from farm5 to mac (make sure the space after run_50401 to documents):
# scp -r ds39@farm5-head2:/lustre/scratch127/mave/sge_analysis/analysis/run_50401 ~/Documents/Sunny/MAVE/MAVE_Rawdata/analysis_quants/


