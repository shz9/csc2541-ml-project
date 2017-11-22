from arch.bootstrap import StationaryBootstrap
from arch.bootstrap import MovingBlockBootstrap
from arch.bootstrap import CircularBlockBootstrap


def stationary_boostrap_method(X, Y, block_size=30, n_samples=50):

    boot_samples = []
    bs = StationaryBootstrap(block_size, X, y=Y)

    for samp in bs.bootstrap(n_samples):
        boot_samples.append((samp[0][0], samp[1]['y']))

    return boot_samples


def moving_block_bootstrap_method(X, Y, block_size=150, n_samples=50):

    boot_samples = []
    bs = MovingBlockBootstrap(block_size, X, y=Y)

    for samp in bs.bootstrap(n_samples):
        boot_samples.append((samp[0][0], samp[1]['y']))

    return boot_samples


def circular_block_bootstrap_method(X, Y, block_size=80, n_samples=50):

    boot_samples = []
    bs = CircularBlockBootstrap(block_size, X, y=Y)

    for samp in bs.bootstrap(n_samples):
        boot_samples.append((samp[0][0], samp[1]['y']))

    return boot_samples