## CInf-FS: An efficient infinite feature selection method using K-means clustering to partition large feature spaces

### Dependencies and Installation
* python >= 3.8
* numpy
* pyitlib
* sklearn
* tqdm
* PyIFS

1. Clone Repo
```
git clone https://github.com/Sadegh28/CInf-FS.git
```

2. Create Conda Environment
```
conda create --name CInf_FS python=3.8
conda activate CInf_FS
```

3. Install Dependencies
```
pip install pyitlib 
conda install -c conda-forge scikit-learn
conda install -c conda-forge numpy
pip install tqdm
pip install PyIFS
```

### Get Started

Use the following command to rank features of  multiple datasets using default params: 

        python PyIT-MLFS.py    --datasets   d1, d2, ..., dn 

Default Params: 
* Data Path: 'data' folder
* Output Path: 'results' folder
* Num of Clusters: 2
* Num of Selected Features: 50
* The Subranks Merging Coefficient (alpha): 0.8

