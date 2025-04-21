import os
import json
import argparse

from omegaconf import OmegaConf

from tgmm.train import train


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config file path")
parser.add_argument("--device_id", type=int, default=-1, help="device id")


def main(args):
    conf_path = os.path.abspath(args.config)
    if not conf_path.endswith(".yaml"):
        raise ValueError("config file must end with .yaml")
    result_path = conf_path.replace(".yaml", ".results.json")
    cfg = OmegaConf.load(conf_path)
    device_id = args.device_id if args.device_id >= 0 else None
    result = train(cfg, device_id, "")  # Name does not matter
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
