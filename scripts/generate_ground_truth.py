"""
    File: generate_ground_truth.py
    Description: Generate ground truth from the best test result.
    Author: dzdydx (sirly.liu@qq.com)
    All rights reserved.
"""

import json

with open('../results/gt.json','r',encoding='utf8') as f:
    test_data = json.load(f)

ground_truth = {}
for item in test_data.items():
    if item[1] > 0.9:
        ground_truth[item[0]] = 1
    elif item[1] < 0.1:
        ground_truth[item[0]] = 0
    else:
        print(f"File {item[0]} not included, its value is {item[1]}")
        ground_truth[item[0]] = 0.5

with open('../results/ground_truth.json', 'w') as f:
    json.dump(ground_truth, f)

