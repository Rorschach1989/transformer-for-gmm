import os
import json
import argparse

import yaml
import wandb

from tgmm.utils import wandb_profile


wandb.login(key=wandb_profile.api_key, relogin=True)

parser = argparse.ArgumentParser()
parser.add_argument("--project_root", type=str, required=True)


def main(args):

    def _push_one(result_path):
        with open(result_path) as f:
            results = json.load(f)

        dirname = os.path.dirname(result_path)
        name = result_path.split("/")[-1].split(".")[0]
        yaml_path = os.path.join(dirname, f"{name}.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        wandb.init(
            project=wandb_profile.project,
            # entity=wandb_profile.entity,
            name=name,
            config=cfg,
        )

        for record in results:
            step = record.pop("step")
            wandb.log(record, step=step)

        wandb.finish()

    pushed = set()
    root = os.path.abspath(args.project_root)

    while True:
        for file in os.listdir(root):
            if file.endswith(".json") and file not in pushed:
                _push_one(os.path.join(root, file))
                pushed.add(file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
