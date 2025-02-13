import argparse
import concurrent.futures as cf

import torch

from tgmm.utils import (
    HyperParamManager,
    get_device_count,
    gen_name_from_cfg,
)
from tgmm.train import train


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, help="Prefix in all the experiments")
parser.add_argument(
    "--mixture_dim", type=int, nargs="*", help="Dimension of mixture means"
)
parser.add_argument(
    "--n_components_max", type=int, nargs="*", help="Maximum number of components"
)
parser.add_argument(
    "--n_components_min", type=int, nargs="*", help="Minimum number of components"
)
parser.add_argument(
    "--train_batch_size", type=int, nargs="*", help="Task per step for training"
)
parser.add_argument("--n_embd", type=int, nargs="*", help="Hidden size of transformer")
parser.add_argument("--n_layer", type=int, nargs="*", help="Size of transformer")
parser.add_argument(
    "--train_n_sample", type=int, nargs="*", help="Maximum length during training"
)
parser.add_argument(
    "--eval_n_sample", type=int, nargs="*", help="Length during evaluation"
)
parser.add_argument(
    "--num_train_steps", type=int, default=10001, help="Number of training steps"
)


def main(args):
    manager = HyperParamManager()
    n_devices = get_device_count()
    manager.register_field("mixture_dim", args.mixture_dim)
    manager.register_field("n_components_max", args.n_components_max)
    manager.register_field("n_components_min", args.n_components_min)
    manager.register_field("train_batch_size", args.train_batch_size)
    manager.register_field("n_embd", args.n_embd)
    manager.register_field("n_layer", args.n_layer)
    manager.register_field("train_n_sample", args.train_n_sample)
    manager.register_field("eval_n_sample", args.eval_n_sample)
    manager.register_field("num_train_steps", args.num_train_steps)

    def _run(manager, cfg, device_id):
        exp_name = gen_name_from_cfg(cfg)
        manager.dump(cfg)
        eval_results = train(cfg, device_id, f"{args.prefix}_{exp_name}")
        manager.save_results(cfg, eval_results)
        return eval_results

    with cf.ThreadPoolExecutor(max_workers=n_devices) as executor:
        futures = {}
        for i, cfg in enumerate(manager.iter_configs()):
            device_id = None if not torch.cuda.is_available() else i % n_devices
            futures[
                executor.submit(_run, manager, cfg, device_id)
            ] = cfg
        for future in cf.as_completed(futures):
            cfg = futures[future]
            eval_results = future.result()

if __name__ == "__main__":
    main(parser.parse_args())
