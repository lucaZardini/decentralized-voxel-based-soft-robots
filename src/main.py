import argparse

from environments.environment import EnvironmentType
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
    arg_parser.add_argument("--nodes", required=False, type=int, nargs="+", default=[15, 2],
                            help="The nodes of the network.")
    arg_parser.add_argument("--eta", required=False, type=float, default=0.1,
                            help="The eta of the network.")
    arg_parser.add_argument("--robot-structure-path", type=str, required=False,
                            default="../data/robot_structure/worm/default.json",
                            help="The path to the robot structure json file.")
    arg_parser.add_argument("--random-structure", default=False, action="store_true", required=False,
                            help="Generate a random structure.")
    arg_parser.add_argument("--train", default=False, action="store_true", required=False,
                            help="Create a network and train it")
    arg_parser.add_argument("--prune", default=False, action="store_true", required=False,
                            help="Prune the network")
    arg_parser.add_argument("--test", required=False, default=False, action="store_true",
                            help="The path to the dataset")
    arg_parser.add_argument("--weight-path", type=str, required=True,
                            help="The path to the weights of the ABCD parameters, or where to store them")
    arg_parser.add_argument("--generations", type=int, required=False, default=30, help="Number of generations")
    arg_parser.add_argument("--offsprings", type=int, required=False, default=15,
                            help="The offsprings")
    arg_parser.add_argument("--population-size", type=int, required=False, default=4,
                            help="The population size")
    arg_parser.add_argument("--sigma", type=float, required=False, default=1.0,
                            help="The sigma")
    arg_parser.add_argument("--evo-algo-type", type=EvoAlgoType, required=False, default=EvoAlgoType.CMAES,
                            choices=list(EvoAlgoType), help="The evolutionary algorithm to use")
    arg_parser.add_argument("--max_steps", type=float, default=2000, required=False,
                            help="Number of attempts to solve the task per individual before being reset")
    arg_parser.add_argument("--weight_update_steps", type=float, default=150, required=False,
                            help="Number of steps before updating the weights")
    arg_parser.add_argument("--prune_ratio", type=int, default=None, required=False,
                            help="Pruning ratio to apply to the network")
    arg_parser.add_argument("--weight_pruning_time", type=int, default=None, required=False,
                            help="Update weight step where to prune the network")
    arg_parser.add_argument("--multi-processing", default=False, action="store_true", required=False,
                            help="Use multi processing")
    arg_parser.add_argument("--raise-error-in-case-of-loading-structure-path", default=True,
                            action="store_true", required=False, help="Raise error in case of loading structure path")
    arg_parser.add_argument("--display", default=False, action="store_true", required=False,
                            help="Display the environment")

    args = arg_parser.parse_args()

    if not args.train and not args.prune and not args.test:
        raise ValueError("You must specify if you want to train, test or prune an existing network.")
    manager = Manager(args.robot, args.robot_structure_path, args.random_structure,
                      args.raise_error_in_case_of_loading_structure_path, args.env, args.network, args.nodes, args.weight_path,
                      args.evo_algo_type, args.offsprings, args.population_size, args.sigma, args.eta)
    if args.train:
        manager.train(args.generations, args.number_of_attempts, args.max_steps, args.weight_update_steps, args.multi_processing, args.display)
    elif args.prune:
        manager.prune(args.max_steps, args.weight_update_steps, args.prune_ratio, args.weight_pruning_time, args.display)
    elif args.test:
        manager.test(args.max_steps, args.weight_update_steps)


if __name__ == "__main__":
    main()
