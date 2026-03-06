"""Trend/stationarity check along strike and depth."""

import os
import json
import numpy as np
import pandas as pd


def run(output_dir='outputs'):
    data = pd.read_csv(os.path.join(output_dir, 'domain_data.csv'))
    # simple linear trends along x (strike proxy) and z (depth)
    trends = {}
    for col in ['x', 'z']:
        coeff = np.polyfit(data[col].values, data['tgc_pct'].values, 1)
        trends[f'{col}_slope'] = float(coeff[0])
        trends[f'{col}_intercept'] = float(coeff[1])

    with open(os.path.join(output_dir, 'tables', 'stationarity_trends.json'), 'w') as f:
        json.dump(trends, f, indent=2)

    return trends


if __name__ == '__main__':
    run()
