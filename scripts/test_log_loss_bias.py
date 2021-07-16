"""
    File: test_log_loss_bias.py
    Description: Test log loss bias after using generated ground truth.
    Author: dzdydx (sirly.liu@qq.com)
    All rights reserved.
"""
import json, math
from pathlib import Path

gt_path = '../results/ground_truth.json'
with open(gt_path, 'r') as f:
    gt = json.load(f)

def calc_log_loss(target_file):
    with open(target_file, 'r') as f:
        target_json = json.load(f)
    
    log_loss = 0
    for item in target_json.items():
        y_hat = item[1]
        y_gt = gt[item[0]]
        epsilon = 1e-30

        log_loss += -(y_gt * math.log(y_hat + epsilon) + (1 - y_gt) * math.log(1 - y_hat + epsilon))

    return log_loss / 20000.0

results = Path('../results')
for test_result in results.rglob("*.json"):
    if str(test_result)[11:].startswith(('SE', 're')):
        print(f'Calculating log loss of {str(test_result)[11:-5]}: \n{calc_log_loss(test_result)}')
