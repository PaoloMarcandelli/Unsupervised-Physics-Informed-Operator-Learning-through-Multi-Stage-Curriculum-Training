import argparse
from types import SimpleNamespace

def get_params():
    """
    Parse command-line arguments and return a namespace of parameters.
    """
    parser = argparse.ArgumentParser(
        description="Configuration for training and fine-tuning"
    )
    # Dataset and model parameters
    parser.add_argument('--ntrain', type=int, default=16, help='Number of training samples')
    parser.add_argument('--ntest', type=int, default=2, help='Number of test samples')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=20, help='Model width parameter')
    parser.add_argument('--bsize', type=int, default=2, help='Batch divisor for calculating batch_size')
    # Training schedule
    parser.add_argument('--epochs', type=int, default=100, help='Number of Adam epochs')
    parser.add_argument('--raj_epochs', type=int, default=150, help='Number of Adam epochs')
    parser.add_argument('--lbfgs_epochs', type=int, default=80, help='Number of LBFGS epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate for Adam')
    parser.add_argument('--scheduler_step', type=int, default=20, help='Step size for LR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for LR scheduler')
    # Grid and resolution
    parser.add_argument('--sub', type=int, default=4, help='Subsampling factor for input data')
    parser.add_argument('--S', type=int, default=64, help='Spatial resolution S')
    parser.add_argument('--T_in', type=int, default=10, help='Input sequence length')
    parser.add_argument('--T', type=int, default=10, help='Prediction sequence length')
    parser.add_argument('--step', type=int, default=1, help='Time step for iteration')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity nu')
    parser.add_argument('--rf', type=int, default=8, help='Resolution factor for super-resolution')
    parser.add_argument('--orders_v', type=int, nargs=2, default=[2,2], help='Spline orders for velocity potential')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of random field samples per batch')
    # Derived parameters are computed after parsing
    args = parser.parse_args()

    # Compute derived values
    args.batch_size = args.ntrain // args.bsize

    return args
