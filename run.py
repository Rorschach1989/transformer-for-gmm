import os
import argparse
import concurrent.futures as cf
import multiprocessing as mp

from tgmm.utils import (
    HyperParamManager,
    get_device_count,
    gen_name_from_cfg,
)
from tgmm.train import train
from tgmm.logger import logger, log_exception_with_traceback


parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs_per_device", default=1, type=int, help="Number of jobs per device")
parser.add_argument("--prefix", type=str, help="Prefix in all the experiments")
parser.add_argument(
    "--recover_from", type=str, default=None, help="Recover from a local directory"
)
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
    "--eval_n_sample", type=str, nargs="*", help="Length during evaluation, seperated by comma"
)
parser.add_argument(
    "--num_train_steps", type=int, default=10001, help="Number of training steps"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-3, help="Number of training steps"
)
parser.add_argument(
    "--weight_decay", type=float, default=0., help="Number of training steps"
)


def _run(manager, cfg, device_queue, exp_name):
    device_id = device_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    logger.info(f"Process {os.getpid()} starting task {exp_name} on GPU {device_id}")
    try:
        manager.dump(cfg)
        eval_results = train(cfg, 0, exp_name)
        manager.save_results(cfg, eval_results)
        return eval_results
    except Exception as e:
        log_exception_with_traceback(logger)
        return {}
    finally:
        device_queue.put(device_id, timeout=1)
        logger.info(f"Process {os.getpid()} has finished task {exp_name} on GPU {device_id}")


def main(args):
    manager = HyperParamManager(args.recover_from)
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
    manager.register_field("learning_rate", args.learning_rate)
    manager.register_field("weight_decay", args.weight_decay)

    with mp.Manager() as mp_manager:
        device_queue = mp_manager.Queue()
        for i in range(n_devices):
            for _ in range(args.n_jobs_per_device):
                device_queue.put(i)

        max_workers = args.n_jobs_per_device * n_devices
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, cfg in enumerate(manager.iter_configs()):
                exp_name = gen_name_from_cfg(cfg)
                if manager.result_exists(cfg):
                    continue
                # device_id = None if not torch.cuda.is_available() else i % n_devices
                futures.append(
                    executor.submit(_run, manager, cfg, device_queue, exp_name)
                )
            for future in cf.as_completed(futures):
                future.result()
            logger.info("All tasks done")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(parser.parse_args())
