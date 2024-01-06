import argparse

from environments.type import EnvironmentType
from evolutionary_algorithm.evo_alg import EvoAlgoType
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
    arg_parser.add_argument("--weight-path", type=str, required=True,
                            help="The path to the weights of the ABCD parameters, or where to store them")
    arg_parser.add_argument("--generations", type=int, required=False, default=10, help="Number of generations")
    arg_parser.add_argument("--offsprings", type=int, required=False, default=20,
                            help="The offsprings")
    arg_parser.add_argument("--population-size", type=int, required=False, default=4,
                            help="The population size")
    arg_parser.add_argument("--sigma", type=float, required=False, default=1.0,
                            help="The sigma")
    arg_parser.add_argument("--evo-algo-type", type=EvoAlgoType, required=False, default=EvoAlgoType.CMAES,
                            choices=list(EvoAlgoType), help="The evolutionary algorithm to use")
    arg_parser.add_argument("--multi-processing", default=False, action="store_true", required=False,
                            help="Use multi processing")

    args = arg_parser.parse_args()

    if not args.train and not args.prune:
        raise ValueError("You must specify if you want to train or prune an existing network.")
    manager = Manager(args.robot, args.robot_structure_path, args.random_structure,
                      args.raise_error_in_case_of_loading_structure_path, args.env, args.network, args.nodes, args.weight_path,
                      args.evo_algo_type, args.offsprings, args.population_size, args.sigma, args.eta)
    if args.train:
        pass
    elif args.prune:
        # manager.test(args.dataset, args.operation, args.digit_len, args.digit_len, args.weight_path, args.network, args.logic_file)
        pass


if __name__ == "__main__":
    main()
