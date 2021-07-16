"""
    TODO: Unfinished

    File: linear_fusion.py
    Description: Search for the coefficients that produce the lowest log loss
    Author: dzdydx (sirly.liu@qq.com)
    All rights reserved.

    Usage:
        python linear_fusion.py output1.json [output2.json] ... [-s 0.01]
"""

import json, math

gt_path = '../results/ground_truth.json'

class Fusion:
    def __init__(self, stride, *files):
        self.stride = stride
        self.files = files
        self.njsons = len(jsons)

        for path in jsons:
            with open(path, 'r') as f:
                self.jsons[path] = json.load(f)

    def _generate_weights(self, stride, njsons, weights):
        weights = []
        if njsons == 2:
            w = 0
            while(w <= 1):
                weights.append([w, 1 - w])
                w += stride
            return weights

    def fuse(self):
        weights = self._generate_weights(self.stride, self.njsons, [])
        result = self.jsons
        for w in weights:
            pass
        pass

    pass


def fusion(stride, *json):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target JSON files to fuse", nargs='?')
    parser.add_argument('-s', '--stride', help='searching stride of the coefficients', default=0.01)
    args = parser.parse_args()

    result, weights = fusion(args.stride, args.target)
    with open(fusion_result_path, 'w') as f:
        json.dump(result, f)
        print(f'Result saved as {fusion_result_path}, coeffients are {i for i in weights}')