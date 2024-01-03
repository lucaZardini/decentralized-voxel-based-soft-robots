import argparse

from environments.type import EnvironmentType
from manager import Manager
from networks.type import NetworkType
from robots.robot import RobotType


def main():
    arg_parser = argparse.ArgumentParser(description="Entrypoint of decentralized voxel based soft robots.")

    arg_parser.add_argument("-r", "--robot", required=True, type=RobotType, choices=list(RobotType),
                            help="The robot you want to use.")
    arg_parser.add_argument("-e", "--env", required=False, type=EnvironmentType, choices=list(EnvironmentType),
                            default=EnvironmentType.WALKING_FLAT, help="The environment you want to use.")
    arg_parser.add_argument("-n", "--network", required=False, type=NetworkType, choices=list(NetworkType),
                            default=NetworkType.HNN, help="The network you want to use.")
    arg_parser.add_argument("--nodes", required=False, type=int, nargs="+", default=[8, 5, 2],
                            help="The nodes of the network.")
    arg_parser.add_argument("--eta", required=False, type=float, default=0.1,
                            help="The eta of the network.")
    arg_parser.add_argument("--robot-structure-path", type=str, required=False,
                            default="data/robot_structure/worm/default.json",
                            help="The path to the robot structure json file.")
    arg_parser.add_argument("--random-structure", default=False, action="store_true", required=False,
                            help="Generate a random structure.")
    arg_parser.add_argument("--train", default=False, action="store_true", required=False,
                            help="Create a network and train it")
    arg_parser.add_argument("--prune", default=False, action="store_true", required=False,
                            help="Prune the network")
    # arg_parser.add_argument("--weight-path", type=str, required=True,
    #                         help="The path to the weights of the ABCD parameters, or where to store them")
    arg_parser.add_argument("--generations", type=int, required=False, default=10, help="Number of generations")
    arg_parser.add_argument("--individuals", type=int, required=False, default=20,
                            help="The individuals per generation")

    args = arg_parser.parse_args()

    if not args.train and not args.prune:
        raise ValueError("You must specify if you want to train or prune an existing network.")
    manager = Manager(args.robot, args.robot_structure_path, args.random_structure,
                      args.raise_error_in_case_of_loading_structure_path, args.env, args.network, args.nodes,
                      args.eta)
    if args.train:
        pass
        # manager.train(
        #     dataset_type=args.dataset,
        #     operation_type=args.operation,
        #     addition_train_length=args.digit_len,
        #     addition_test_length=args.digit_len,
        #     weight_out_path=args.weight_path,
        #     network_type=args.network,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     learning_rate=args.learning_rate,
        #     logic_file=args.logic_file,
        #     limit_train_dataset=args.limit_train
        # )
    elif args.prune:
        # manager.test(args.dataset, args.operation, args.digit_len, args.digit_len, args.weight_path, args.network, args.logic_file)
        pass


if __name__ == "__main__":
    main()
