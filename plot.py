#!/usr/bin/env python3

import polars as pl
import matplotlib.pyplot as plt

from atompaint.analysis.plot_metrics import (
        load_tensorboard_logs, plot_training_metrics,
        extract_hparams, pick_metrics,
)
from pathlib import Path

df = load_tensorboard_logs([Path(__file__).parent])

df = extract_hparams(df, r'''(?x)
        (?<algorithm>fourier|gated)_
        (?<function>.*)/
        version_\d+
''')

# My naming convention changed half-way through, so fix that here.
df = df.with_columns(
        pl.col('function').replace(dict(
            hard_shrink='hardshrink',
            leaky_hard_shrink='leaky_hardshrink',
            first_hermite='hermite',
        )),
)

# The hardshrink results so bad that they make the rest of the plots hard to 
# read, so just exclude them.
df = df.filter(
        pl.col('function') != 'leaky_hardshrink',
        pl.col('function') != 'hardshrink',
        pl.col('algorithm') == 'gated',
)

df = df.with_columns(
        function_family=pl.col('function').replace(dict(
            relu='rectified',
            elu='rectified',
            silu='rectified',
            gelu='rectified',
            mish='rectified',

            hardtanh='sigmoid',
            softsign='sigmoid',
            sigmoid='sigmoid',
            tanh='sigmoid',

            hardshrink='linear',
            leaky_hardshrink='linear',
            softshrink='linear',
            tanhshrink='linear',

            hermite='hermite',
        )),

        smooth=pl.col('function').replace(dict(
            relu='no',
            elu='yes',
            silu='yes',
            gelu='yes',
            mish='yes',

            hardtanh='no',
            softsign='yes',
            sigmoid='yes',
            tanh='yes',

            hardshrink='no',
            leaky_hardshrink='no',
            softshrink='no',
            tanhshrink='yes',

            hermite='yes',
        )),
)

metrics = pick_metrics(df, include_train=True)
hparams = 'algorithm', 'function', 'function_family', 'smooth'

plot_training_metrics(df, metrics, hparams)
#plt.savefig('compare_nonlinearity.svg')
plt.show()
