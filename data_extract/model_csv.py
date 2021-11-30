import numpy as np
import pickle
import optparse
import os

regions_all = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY']

# Get the command line arguments
parser = optparse.OptionParser()
parser.add_option('-m', '--model', dest='model', help='Model Name')

# Parse the command line arguments
(options, args) = parser.parse_args()
model = options.model

dt = []
for i in range(2, 5):
    with open(os.path.join("model_preds", f"{model}_preds_{i}.pkl"), "rb") as fl:
        preds = pickle.load(fl)
        dt.append(preds)

target_file = os.path.join("model_preds", model + "_pred.csv")

with open(target_file, 'w') as f:
    f.write("epiweek,region,target,value\n")
    for i in range(1, 33):
        for j, r in enumerate(regions_all):
            for w in range(2, 5):
                f.write(f"{i},{r},{w} weeks ahead,{dt[w-2][j][i-1]}\n")


