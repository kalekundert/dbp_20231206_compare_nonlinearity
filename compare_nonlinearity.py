#!/usr/bin/env python3

"""\
Compare different equivariant nonlinearities.

Usage:
    compare_nonlinearity.py [<hparams>] [-d]

Arguments:
    <hparams>
        The name of the hyperparameters to use.  If not specified, print out a
        list of all the valid hyperparameter names.

Options:
    -d --debug
        If true, run only 10 steps and don't save any results.
"""

import torch
import torch.nn.functional as F

from atom3d_menagerie.predict import RegressionModule, get_trainer
from atom3d_menagerie.hparams import label_hparams, require_hparams
from atom3d_menagerie.data.smp import get_default_smp_data
from atom3d_menagerie.models.escnn import (
        EquivariantCnn,
        conv_bn_norm,
        conv_bn_gated,
        conv_bn_fourier,
        pool_conv,
        invariant_conv,
        linear_relu_dropout,
)
from atompaint.nonlinearities import first_hermite as hermite, leaky_hard_shrink
from atompaint.encoders.layers import (
        make_trivial_field_type, make_fourier_field_types,
)
from escnn.gspaces import rot3dOnR3
from torch.optim import Adam
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

@dataclass
class Norm:
    # Don't want to bother making norm nonlinearities accept arbitrary 
    # functions, so I'm just going to limit myself to the currently supported 
    # ones.
    nonlinearity: str

    def __str__(self):
        return f'norm_{self.nonlinearity}'

    def make_conv_factory(self, _):
        if self.nonlinearity == 'squash':
            nonlinearity = 'squash'
            bias = False
        else:
            nonlinearity = f'n_{self.nonlinearity}'
            bias = True

        return partial(
                conv_bn_norm,
                function=nonlinearity,
                bias=bias,
        )

@dataclass
class Gated:
    nonlinearity: Callable[[torch.Tensor], torch.Tensor]

    def __str__(self):
        return f'gated_{self.nonlinearity.__name__}'

    def make_conv_factory(self, _):
        return partial(
                conv_bn_gated,
                function=self.nonlinearity,
        )

@dataclass
class Fourier:
    nonlinearity: Callable[[torch.Tensor], torch.Tensor]
    extra_freqs: int = 0

    def __str__(self):
        name = f'fourier_{self.nonlinearity.__name__}'
        if self.extra_freqs:
            name += f'_extra_{self.extra_freqs}'
        return name

    def make_conv_factory(self, so3):
        grid = so3.grid('thomson_cube', N=4)
        return partial(
                conv_bn_fourier,
                function=self.nonlinearity,
                ift_grid=grid,
                extra_freqs=self.extra_freqs,
        )

HPARAMS = label_hparams(
        str,

        Norm('relu'),
        Norm('sigmoid'),
        Norm('softplus'),
        Norm('squash'),

        Gated(F.sigmoid),

        Gated(F.relu),
        Gated(F.elu),
        Gated(F.silu),
        Gated(F.gelu),
        Gated(F.mish),

        Fourier(F.relu),
        Fourier(F.elu),
        Fourier(F.silu),
        Fourier(F.gelu),
        Fourier(F.mish),

        Fourier(F.hardtanh),
        Fourier(F.tanh),
        Fourier(F.softsign),

        Fourier(F.softshrink),
        Fourier(F.hardshrink),
        Fourier(leaky_hard_shrink),
        Fourier(F.tanhshrink),
        
        Fourier(hermite),
)

def make_escnn_model(hparams):
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    so2_z = False, -1
    L = 2

    field_types = [
            make_trivial_field_type(gspace, 5),
            *make_fourier_field_types(
                gspace,
                channels=[1, 2, 4],
                max_frequencies=L,
            ),
            *make_fourier_field_types(
                gspace,
                channels=[28],
                max_frequencies=L,
                subgroup_id=so2_z,
            ),
    ]

    return EquivariantCnn(
            field_types=field_types,
            conv_factory=hparams.make_conv_factory(so3),
            pool_factory=pool_conv,
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_conv,
                out_channels=256,
                kernel_size=3,
            ),
            mlp_channels=[512],
            mlp_factory=partial(
                linear_relu_dropout,
                drop_rate=0.25,
            ),
    )


if __name__ == '__main__':
    import docopt

    args = docopt.docopt(__doc__)
    hparams_name, hparams = require_hparams(args['<hparams>'], HPARAMS)

    model = make_escnn_model(hparams)
    data = get_default_smp_data()

    trainer = get_trainer(
            Path(hparams_name),
            max_epochs=50,
            fast_dev_run=args['--debug'] and 10,
    )
    model = RegressionModule(model, Adam)
    trainer.fit(model, data)
