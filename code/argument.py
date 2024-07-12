def dqn_arguments(parser):

    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')
    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--episodes", default=int(400), type=int)
    parser.add_argument('--epsilon', type=float, default=0.95)
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size for training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--target_update', type=int, default=10) 
    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)
    # add
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')

    return parser
