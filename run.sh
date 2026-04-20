conda init
conda activate dl
python Segy2H5.py
python split_core.py sample 0.5 random
python split_core.py sample 0.3 line_recv