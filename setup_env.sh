#git clone https://github.com/tomole444/Cutie/
#cd Cutie
source /home/$USER/anaconda3/etc/profile.d/conda.sh
conda create -n cutie python==3.10.12
conda activate cutie
pip install -e .
pip install -e .

#download training weights
python scripts/download_models.py