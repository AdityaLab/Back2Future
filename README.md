# Back2Future: Leveraging Backfill Dynamics forImproving Real-time Predictions in Future

Link to paper: https://arxiv.org/abs/2106.04420

## Setup

First install Anaconda. The dependencies are listed in `environment.yml` file. Make sure you make changes to version of `cudatoolkit` if applicable.

Then run the following commands:

```bash
conda env create --prefix ./envs/backfill --file environment.yml
source activate ./envs/backfill
```

## Directory structure

```
-data_extract
	- create_table.py -> preprocess raw covid dataset as pkl files
- gnnrnn/gnn_model.py -> implrmrntation of B2F
- model_preds -> folder containing predictions of hub models
- results -> stores predictions
- covid_utils.py -> utility functions to extract bseqs
- train_bseqenc.py -> Pre-train BeseqEnc
- train_b2f.py -> Train B2F for given week and model prediction history and infer predictions
```

## Dataset

The dataset is at `covid_data` folder. It contains csv file for each week that contains revised values for all signals from all previous weeks and current week. For example `covid_data/covid-hospitalization-all-state-merged_vEW202030.csv` contains revised dataset observed on week 30.

## Training and Predictions

### Extract data
Run `extract.sh` to extract backfill values with missing values filled (as described in supplementary) at `saves`.

### Pretrain BseqEnc

Run:

```bash
python train_bseqenc.py -l <current_week> -p <epochs> -c <cuda? yes/no> -m <minimum lenght of bseq> -n <experiment name>
```

This will store a trained bseqenc at `saves\<experiment name>_rev_model.pth`

### Fine-tune B2F

Run:

```bash
python train_b2f.py -l <current_week> -e <epochs> -a <weeks ahead> -c <cuda? yes/no> -n <experiment name>
```
This will provide predictions in a dictionary at `saves\<experiment name>_pred.pkl` in form of a dictionary like:

```python
{'Expt name': 'week_40_1',
 'Weeks ahead': 2,
 'regions': ['CA', 'DC', 'FL', 'GA', 'IL', 'NY', 'TX', 'WA'],
 'current week': 40,
 'forecast': array([410.17530001,   4.82940001, 610.0748    , 209.7412    ,
        226.7672    ,  58.44170001, 658.7947    ,  47.6249    ]),
 'refined': array([485.72659425,   5.42677917, 614.71402962, 216.20428756,
        213.2565347 ,  68.36743161, 664.00933847,  47.86988136])}
```
The model is stored in `saves` folder with files:

- `saves\<experiment name>_fine_rev_model.pth`
- `saves\<experiment name>_fine_bias_encoder.pth`
- `saves\<experiment name>_fine_refiner.pth`

**Note:** For example of a single run of full pipeline see `example.sh`
