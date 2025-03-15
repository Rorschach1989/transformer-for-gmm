import os
import json
import argparse

import yaml
import wandb

from tgmm.utils import wandb_profile


wandb.login(key=wandb_profile.api_key, relogin=True)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_prefix", type=str, required=True)
parser.add_argument("--project_root", type=str, required=True)
parser.add_argument("--wandb_project", type=str, default="TGMM")


def main(args):

    def _get_existent_names_and_ids():
        api = wandb.Api()
        nams_and_ids = {}
        for run in api.runs(f"{api.default_entity}/TGMM"):
            nams_and_ids[run.name] = run.id
        return nams_and_ids

    root = os.path.abspath(args.project_root)
    pushed = set(_get_existent_names_and_ids())

    def _push_one(result_path, append_prefix=True):
        with open(result_path) as f:
            results = json.load(f)

        dirname = os.path.dirname(result_path)
        name = result_path.split("/")[-1].split(".")[0]
        yaml_path = os.path.join(dirname, f"{name}.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        if append_prefix:
            name = f"{args.exp_prefix}.{name}"
        if name in pushed:
            return

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
        return name

    while True:
        for file in os.listdir(root):
            if file.endswith(".json"):
                exp_name = _push_one(os.path.join(root, file))
                pushed.add(exp_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
