"""
    File: calc_log_loss.py
    Description: Calculate log loss between given test result and ground truth.
    Author: dzdydx (sirly.liu@qq.com)
    All rights reserved.

    Usage:
        python calc_log_loss.py [path_to_target_JSON_file]
"""
import argparse, json, math

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

        log_loss += -(y_gt * math.log(y_hat + epsilon) + (1 - y_gt - epsilon) * math.log(1 - y_hat))

    return log_loss / 20000.0

parser = argparse.ArgumentParser()
parser.add_argument("target", help="target JSON file to calculate log loss")
args = parser.parse_args()

if __name__ == '__main__':
    print(calc_log_loss(args.target))
