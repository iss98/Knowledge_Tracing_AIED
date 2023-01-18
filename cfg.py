import argparse
import torch
from model import MODEL_LIST

def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model", type = str, default = "DKT", choices=MODEL_LIST, help = "model type"
    )
    parser.add_argument(
        "--dscor", type = str, choices = ["high", "low", "indep"], default = "high", help="dataset assumption(correlation)"
    )
    parser.add_argument(
        "--dsitem", type = str, choices = ["fixed", "rand"], default = "fixed", help="dataset assumption(item)"
    )
    parser.add_argument(
        "--dstime", type = str, choices = ["fixed", "rand"], default = "fixed", help="dataset assumption(time)"
    )
    parser.add_argument(
        "--addds", type = bool, default = False, help="add additional data or not"
    )
    parser.add_argument(
        "--prjname", type = str, default = "Knowledge Tracing", help = "name of the project for wandb"
    )
    parser.add_argument(
        "--item", type = int, help = "number of items"
    )
    parser.add_argument(
        "--edim", type = int, default = 64, help = "dimension of embedding"
    )
    parser.add_argument(
        "--hdim", type = int, default = 64, help = "dimension of hidden layer"
    )
    parser.add_argument(
        "--dr", type = float, default = 0.5 , help = "dropout rate of dropout layer"
    )
    parser.add_argument(
        "--heads", type = str, default = 8 , help = "number of heads for attention layer"
    )
    parser.add_argument(
        "--cnum", type = str, default = 13 , help = "number of latent concepts"
    )
    parser.add_argument(
        "--lam", type = float, default = 0.5 , help = "lambda for RKT"
    )
    parser.add_argument(
        "--device", type = str, default = "cuda:0" if torch.cuda.is_available() else "cpu", help = "GPU"
    )
    parser.add_argument(
        "--testsplit", type = float, default = 0.2, help = "split testset"
    )
    parser.add_argument(
        "--valsplit", type = float, default = 0.25, help = "split validset"
    )
    parser.add_argument(
        "--lr", type = float, default = 1e-3, help = "learning rate"
    )
    parser.add_argument(
        "--mt", type = float, default = 0.9, help = "momentum for SGD"
    )
    parser.add_argument(
        "--fac", type = float, default = 0.5, help = "factor for scheduler"
    )
    parser.add_argument(
        "--pat", type = str, default = 5, help = "patience for scheduler"
    )
    parser.add_argument(
        "--thr", type = float, default = 1e-4, help = "threshold for scheduler"
    )
    parser.add_argument(
        "--epoch", type = str, default = 200, help = "number of epochs"
    )
    parser.add_argument(
        "--testevery", type = str, default = 10, help = "test period"
    )
    parser.add_argument(
        "--bs", type = str, default = 64, help = "batch size"
    )
    return parser.parse_args()