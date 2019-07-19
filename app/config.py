import os

data_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data'
raw_path = f"{data_path}/raw"
train_path = f"{data_path}/train"
test_path = f"{data_path}/test"
checkpoint_path = f"{data_path}/checkpoint"
