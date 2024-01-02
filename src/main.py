import argparse

from environments.type import EnvironmentType
from manager import Manager
from robots.robot import RobotType


def main():
    arg_parser = argparse.ArgumentParser(description="Entrypoint of decentralized voxel based soft robots.")

    arg_parser.add_argument("-r", "--robot", required=True, type=RobotType, choices=list(RobotType),
                            help="The robot you want to use.")
    arg_parser.add_argument("-e", "--env", required=True, type=EnvironmentType, choices=list(EnvironmentType),
                            help="The environment you want to use.")
    arg_parser.add_argument("--train", default=False, action="store_true", required=False,
                            help="Create a network and train it")
    arg_parser.add_argument("--prune", default=False, action="store_true", required=False,
                            help="Prune the network")
    arg_parser.add_argument("--weight-path", type=str, required=True,
                            help="The path to the weights of the ABCD parameters, or where to store them")
    arg_parser.add_argument("--generations", type=int, required=False, default=10, help="Number of generations")
    arg_parser.add_argument("--individuals", type=int, required=False, default=20,
                            help="The individuals per generation")

    args = arg_parser.parse_args()

    if not args.train and not args.test and not args.train_and_test:
        raise ValueError("You must specify if you want to train or test the network.")
    manager = Manager()
    # if args.network == NetworkType.DEEP_PROBLOG and not args.logic_file:
    #     raise ValueError("You must specify the logic file for DeepProbLog.")
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
    elif args.test:
        # manager.test(args.dataset, args.operation, args.digit_len, args.digit_len, args.weight_path, args.network, args.logic_file)
        pass


if __name__ == "__main__":
    main()
