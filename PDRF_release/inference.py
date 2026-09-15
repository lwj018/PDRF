"""Run PDRF inference on the test set and evaluate CSI / HSS / SSIM / MSE.

Loads a best checkpoint (see Exps/*/checkpoints/ckpt-best*.pt or the
pretrained files in ../PDRF/bestpth) and generates forecasts with the
fixed 5-step ODE sampler (paper Sec. 4.1).

    python inference.py --dataset shanghai \
        --ckpt_milestone /root/autodl-tmp/PDRF/bestpth/ckpt-best-shanghai.pt
"""
import os
import os.path as osp
import time
import argparse
import logging
import json
import yaml
from tqdm import tqdm
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs

from datasets.get_datasets import get_dataset
from utils.tools import print_log
from utils.metrics import Evaluator
from pdrf.model import build_pdrf
from flow.crf import CRF

crf = CRF()

os.environ["WANDB_API_KEY"] = "YOUR_WANDB"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# frames_out per dataset (paper Sec. 4.1: 5 -> 20; CIKM uses 5 -> 10)
FRAMES_OUT = {'shanghai': 20, 'meteo': 20, 'sevir': 20, 'cikm': 10}


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help='experiment seed')
    parser.add_argument("--exp_dir", type=str, default='basic_exps', help="experiment directory")
    parser.add_argument("--exp_note", type=str, default=None, help="additional note for experiment")

    parser.add_argument("--dataset", type=str, default='shanghai',
                       choices=['shanghai', 'cikm', 'meteo', 'sevir'], help="dataset name")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--stride", type=int, default=13, help="dataset stride")
    parser.add_argument("--seq_len", type=int, default=25, help="sequence length sampled from dataset")
    parser.add_argument("--frames_in", type=int, default=5, help="number of frames to input")
    parser.add_argument("--frames_out", type=int, default=None,
                       help="number of frames to output (default: 20, CIKM: 10)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
    parser.add_argument("--ckpt_milestone", type=str, required=True,
                       help="checkpoint path (.pt) to evaluate")
    parser.add_argument("--use_ema", action="store_true",
                       help="evaluate with EMA weights instead of online weights")
    parser.add_argument("--max_batches", type=int, default=0,
                       help="only evaluate the first N batches (0 = all)")
    parser.add_argument("--gpu_use", type=str, nargs='+', default=[], help="gpu(s) to use")

    return parser.parse_args()


class InferenceRunner(object):

    def __init__(self, args):
        self.args = args
        if args.frames_out is None:
            args.frames_out = FRAMES_OUT[args.dataset]
        self._preparation()

        project_config = ProjectConfiguration(project_dir=self.exp_dir, logging_dir=self.log_path)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))

        self.accelerator = Accelerator(
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs, process_kwargs],
            mixed_precision='no',
        )

        self._load_data()
        self.test_loader = self.accelerator.prepare(self.test_loader)
        self._build_model()
        self.model = self.accelerator.prepare(self.model)

        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def device(self):
        return self.accelerator.device

    def _preparation(self):
        set_seed(self.args.seed)
        self.model_name = 'PDRF'
        self.exp_name = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}_inference"

        cur_dir = os.path.dirname(os.path.abspath(__file__))

        self.exp_dir = osp.join(cur_dir, 'Inference_Results', self.args.exp_dir, self.exp_name)
        self.test_path = osp.join(self.exp_dir, 'test_samples')
        self.log_path = osp.join(self.exp_dir, 'logs')

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        yaml.dump(self.args.__dict__, open(osp.join(self.exp_dir, 'params.yaml'), 'w'))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'inference_log.log')),
                logging.StreamHandler()
            ]
        )

    def _load_data(self):
        _, _, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
            stride=self.args.stride,
        )

        self.visiual_save_fn = color_save_fn
        self.thresholds = THRESHOLDS
        self.scale_value = PIXEL_SCALE

        if self.args.dataset != 'sevir':
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)

        print_log(f"test_data: {len(self.test_loader)}", self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}", self.is_main)

    def _build_model(self):
        print_log("Build PDRF model!", self.is_main)
        model = build_pdrf(
            frames_in=self.args.frames_in, frames_out=self.args.frames_out
        ).cuda()
        self.model = model
        total = sum([param.nelement() for param in self.model.parameters()])
        print_log("Main Model Parameters: %.2fM" % (total / 1e6), self.is_main)

    def load(self, milestone):
        device = self.accelerator.device
        data = torch.load(milestone, map_location=device)
        print_log(f"Load checkpoint {milestone}.", self.is_main)

        model = self.accelerator.unwrap_model(self.model)
        if self.args.use_ema:
            # checkpoint stores both online_model.* and ema_model.* keys
            ema_sd = {k[len('ema_model.'):]: v
                      for k, v in data['ema'].items() if k.startswith('ema_model.')}
            missing, unexpected = model.load_state_dict(ema_sd, strict=True)
            weight_name = 'ema_model (EMA weights)'
        else:
            missing, unexpected = model.load_state_dict(data['model'], strict=True)
            weight_name = 'model (online weights)'
        print_log(f"Weights loaded strictly: {weight_name}", self.is_main)
        assert len(missing) == 0 and len(unexpected) == 0, (missing, unexpected)
        self.model = self.accelerator.prepare(model)

    def _get_seq_data(self, batch):
        return batch[:, :self.args.frames_out + self.args.frames_in]  # [B, T, C, H, W]

    @torch.no_grad()
    def _sample_batch(self, batch):
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:, :frame_in], radar_batch[:, frame_in:]

        z = torch.randn(radar_input.shape[0], self.args.frames_out,
                        self.args.img_size, self.args.img_size,
                        requires_grad=False).to(self.device)
        condition = radar_input.squeeze(2)
        pred = crf.sample(self.model, z, condition, device=self.device, progress=False)
        radar_pred = pred.unsqueeze(2)

        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()
        radar_input = self.accelerator.gather(radar_input).detach().cpu().numpy()

        return radar_gt, radar_pred, radar_input

    def inference(self):
        save_dir = osp.join(self.test_path, "inference_results")
        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()

        eval = Evaluator(
            seq_len=self.args.frames_out,
            value_scale=self.scale_value,
            thresholds=self.thresholds,
            save_path=save_dir,
        )

        global_img_count = 0
        print_log("Starting inference...", self.is_main)

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Inference',
                                               disable=not self.is_main)):
            if self.args.max_batches > 0 and batch_idx >= self.args.max_batches:
                break

            radar_ori, radar_recon, radar_input = self._sample_batch(batch)

            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)

                batch_size = radar_ori.shape[0]
                for i in range(batch_size):
                    img_dir = osp.join(save_dir, f"sample_{global_img_count:06d}")
                    vil_out_dir = osp.join(img_dir, "vil")
                    vil_in_dir = osp.join(img_dir, "vil_in")
                    os.makedirs(vil_out_dir, exist_ok=True)
                    os.makedirs(vil_in_dir, exist_ok=True)

                    self.visiual_save_fn(radar_recon[i], radar_ori[i], vil_out_dir, data_type='vil')
                    self.visiual_save_fn(radar_input[i], radar_input[i], vil_in_dir, data_type='vil')
                    global_img_count += 1

            self.accelerator.wait_for_everyone()

        if self.is_main:
            res = eval.done(is_main_process=self.is_main)
            print_log(f"Inference Results: {res}")
            print_log(f"Total samples processed: {global_img_count}")
            print_log(f"Results saved to: {save_dir}")
            print_log("=" * 50)

            result_summary = {
                "total_samples": global_img_count,
                "metrics": res,
                "save_directory": save_dir,
                "checkpoint_used": self.args.ckpt_milestone,
                "use_ema": self.args.use_ema,
            }
            with open(osp.join(save_dir, "inference_summary.json"), 'w') as f:
                json.dump(result_summary, f, indent=2)

        print_log("Inference completed successfully!", self.is_main)


def main():
    args = create_parser()
    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not osp.isfile(args.ckpt_milestone):
        raise ValueError(f"Checkpoint not found: {args.ckpt_milestone}")

    runner = InferenceRunner(args)
    runner.inference()


if __name__ == '__main__':
    main()
