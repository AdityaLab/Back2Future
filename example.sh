# Extract data
bash ./extract.sh 
# Pre-train bseqenc for week 40
python train_b2f.py -l 40 -n week_40_1
# Train b2f for gtds 2 week ahead
python train_b2f.py -l 40 -n week_40_1 -e 2000