#Open VSCode
#New Terminal -> PowerShell

# create env

conda create -n env python=3.7 anaconda

# activate env

conda activate env

# install lib

pip install -r requirements.txt

# run demo

streamlit run main.py
