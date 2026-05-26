import os
import os.path as osp
import math
import time
import argparse
import logging
import yaml
from tqdm import tqdm
from datetime import timedelta
import numpy as np

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs

from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
cuda_idx = 3
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)

from datasets.get_datasets import get_dataset
from utils.tools import print_log, cycle

# percpcast
from models.percpcast.denoising_diffusion import GaussianDiffusion
from models.percpcast.unet import Unet
from pysteps_extrapolation2 import PySTEPSExtrapolationWrapper
# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "YOUR_WANDB"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def create_parser():
    parser = argparse.ArgumentParser()

    # --------------- Basic ---------------
    parser.add_argument('--backbone', type=str, default='percpcast')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_dir", type=str, default='basic_exps')
    parser.add_argument("--exp_note", type=str, default='exp')

    # --------------- Dataset ---------------
    parser.add_argument("--dataset", type=str, default='shanghai')
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=13)
    parser.add_argument("--img_channel", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--frames_in", type=int, default=5)
    parser.add_argument("--frames_out", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)

    # --------------- Optimizer ---------------
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-beta1", type=float, default=0.90)
    parser.add_argument("--lr-beta2", type=float, default=0.95)
    parser.add_argument("--l2-norm", type=float, default=0.0)
    parser.add_argument("--ema_rate", type=float, default=0.95)
    parser.add_argument("--scheduler", type=str, default='cosine', choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, default='no')
    parser.add_argument("--grad_acc_step", type=int, default=1)

    # --------------- Training ---------------
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--training_steps", type=int, default=1)
    parser.add_argument("--ckpt_milestone", type=str, default=None)
    parser.add_argument("--early_stop", type=int, default=10)

    # --------------- Additional ---------------
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--valid_limit", action="store_true")
    parser.add_argument("--vlnum", type=int, default=10)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--wandb_state", type=str, default='disabled')
    parser.add_argument("--gpu_use", type=str, nargs='+', default=[])
    parser.add_argument("--res_opt", action="store_true")

    args = parser.parse_args()
    return args


class Runner(object):
    def __init__(self, args):
        self.args = args
        self._preparation()

        self.max_csi, self.best_step = 0.0, 0

        project_config = ProjectConfiguration(project_dir=self.exp_dir, logging_dir=self.log_path)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))

        self.accelerator = Accelerator(
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs, process_kwargs],
            mixed_precision=self.args.mixed_precision,
            log_with='wandb'
        )

        self.accelerator.init_trackers(
            project_name=self.exp_name,
            config=self.args.__dict__,
            init_kwargs={"wandb": {"mode": self.args.wandb_state}}
        )

        print_log(f"Using device: {self.device}", self.is_main)
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
        print_log(self.accelerator.state, self.is_main)

        self._load_data()
        self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.train_loader, self.valid_loader, self.test_loader
        )

        self._build_model()
        self._build_optimizer()

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)

        self.cur_step, self.cur_epoch = 0, 0
        self.cdf_exp = None

        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def device(self):
        return self.accelerator.device

    def _preparation(self):
        set_seed(self.args.seed)
        self.model_name = self.args.backbone
        self.exp_name = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.exp_dir = osp.join(cur_dir, 'Exps_percpcast_xpred', self.args.exp_dir, self.exp_name)
        self.ckpt_path = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path = osp.join(self.exp_dir, 'test_samples')
        self.log_path = osp.join(self.exp_dir, 'logs')

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        yaml.dump(self.args.__dict__, open(osp.join(self.exp_dir, 'params.yaml'), 'w'))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.FileHandler(osp.join(self.log_path, 'log.log'))]
        )

    def _load_data(self):
        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
            stride=self.args.stride,
        )

        self.visiual_save_fn = color_save_fn
        self.scale_value = PIXEL_SCALE

        # ===== 你要求：不用数据集不同阈值 -> 固定阈值（Evaluator 默认 [20,30,35,40]）=====
        self.thresholds = THRESHOLDS

        if self.args.dataset != 'sevir':
            # 你原逻辑：非 sevir 用“大 batch”做累积（我保持不动）
            self.train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.args.batch_size * self.args.grad_acc_step,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)

        print_log(
            f"train: {len(self.train_loader)}, valid: {len(self.valid_loader)}, test: {len(self.test_loader)}",
            self.is_main
        )
        print_log(
            f"Pixel Scale: {PIXEL_SCALE}, Use Fixed Thresholds: {self.thresholds}",
            self.is_main
        )

    def _build_model(self):
        print_log("Build Model!", self.is_main)

        if self.args.backbone == 'percpcast':
            denoise_model = Unet(dim=64, channels=self.args.img_channel, pred_length=self.args.frames_out)
            model = GaussianDiffusion(
                denoise_fn=denoise_model,
                channels=self.args.img_channel,
                pred_mode="noise",
                aux_loss=False,
                T_in=self.args.frames_in,
                T_out=self.args.frames_out,
                image_size=self.args.img_size,
                device=self.device,
            )
        else:
            raise NotImplementedError("This file is configured for percpcast only (as requested).")

        self.model = model
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)

        if self.is_main:
            total = sum([p.nelement() for p in self.model.parameters()])
            print_log(f"Model Params: {total/1e6:.2f}M", self.is_main)

    def _build_optimizer(self):
        num_steps_per_epoch = len(self.train_loader)
        num_epoch = math.ceil(self.args.training_steps / max(num_steps_per_epoch, 1))
        self.global_epochs = max(num_epoch, self.args.epochs)
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )

        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.global_steps
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.global_steps
            )
        else:
            raise ValueError(f"Invalid scheduler_type: {self.args.scheduler}")

        if self.is_main:
            print_log(f"Epochs: {self.global_epochs}, Steps: {self.global_steps}", self.is_main)

    def save(self, svname=None):
        if not self.is_main:
            return

        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        name = f"ckpt-{self.cur_step}.pt" if svname is None else f"ckpt-{svname}.pt"
        torch.save(data, osp.join(self.ckpt_path, name))
        print_log(f"Saved {name}", self.is_main)

    def load(self, milestone):
        device = self.device
        if isinstance(milestone, str) and milestone.endswith('.pt'):
            data = torch.load(milestone, map_location=device)
            print_log(f"Load checkpoint {milestone}.", self.is_main)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
            print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=True)

        if self.args.res_opt:
            if 'opt' in data:
                self.optimizer.load_state_dict(data['opt'])
            if 'scheduler' in data:
                self.scheduler.load_state_dict(data['scheduler'])
            self.cur_epoch = int(data.get('epoch', 0)) + 1
            self.cur_step = int(data.get('step', 0))

    # ============================
    # batch helper: ensure [B,T,C,H,W]
    # ============================
    def _to_btc_hw(self, batch):
        if isinstance(batch, dict):
            x = batch.get('vil', None)
            if x is None:
                raise ValueError("batch is dict but no 'vil' key")
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        # [B,T,H,W] -> [B,T,1,H,W]
        if x.dim() == 4:
            x = x.unsqueeze(2)

        expected_T = self.args.frames_in + self.args.frames_out
        # [T,B,C,H,W] -> [B,T,C,H,W] (按你旧 Trainer 数据格式兼容)
        if x.dim() == 5 and x.shape[0] == expected_T and x.shape[1] != expected_T:
            x = x.permute(1, 0, 2, 3, 4).contiguous()

        return x.to(self.device)

    # ============================
    # cdf_exp per epoch (Trainer style)
    # ============================
    def _build_cdf_exp(self, epoch: int):
        M = np.arange(self.args.frames_out)
        k = 0.05
        T_start, T_end, t_max = 50, 0.5, 100
        T_exp = T_start * (T_end / T_start) ** (epoch / float(t_max))
        weights = np.exp(k * M)
        weights_exp = weights ** (1.0 / T_exp)
        p_exp = weights_exp / np.sum(weights_exp)
        return np.cumsum(p_exp)

    # ============================
    # REQUIRED: _train_batch (Trainer behavior)
    # ============================
    def _train_batch(self, batch):
        video = self._to_btc_hw(batch)  # [B,T,C,H,W]
        expected_T = self.args.frames_in + self.args.frames_out
        video = video[:, :expected_T]

        model_call = self.model.module if hasattr(self.model, "module") else self.model
        t, loss = model_call(video, self.cdf_exp)  # 旧 Trainer: t, loss = self.model(data, cdf_exp)

        return {"total_loss": loss, "t_index": t.float().mean()}

    @torch.no_grad()
    def _sample_batch(self, batch):
        video = self._to_btc_hw(batch)
        expected_T = self.args.frames_in + self.args.frames_out
        video = video[:, :expected_T]

        init_frames = video[:, :self.args.frames_in]      # [B,Tin,C,H,W]
        gt = video[:, self.args.frames_in:]               # [B,Tout,C,H,W]

        model_sample = self.model.module if hasattr(self.model, "module") else self.model
        pred = model_sample.sample(init_frames, num_of_frames=self.args.frames_out).clamp(0, 1)  # [B,Tout,H,W]
        pred = pred.unsqueeze(2)  # [B,Tout,1,H,W]

        gt = self.accelerator.gather(gt).detach()
        pred = self.accelerator.gather(pred).detach()
        return gt, pred

    def train(self):
        pbar = tqdm(initial=self.cur_step, total=self.global_steps, disable=not self.is_main)

        for epoch in range(self.cur_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()

            # per epoch cdf_exp
            self.cdf_exp = self._build_cdf_exp(epoch)

            for i, batch in enumerate(self.train_loader):
                with self.accelerator.autocast():
                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])

                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                log_dict = {'lr': float(lr)}
                for k, v in loss_dict.items():
                    log_dict[k] = float(v.item()) if torch.is_tensor(v) else float(v)

                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)
                pbar.set_description(f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}")

                if i % 20 == 0:
                    logging.info(f"Epoch {self.cur_epoch} step {i} :: {log_dict}")

                self.ema.update()
                self.cur_step += 1
                pbar.update(1)

            # 每 epoch 评估/保存：用最新 Evaluator
            if self.args.valid:
                cur_csi = self.test_samples(self.cur_step, do_test=False)
                if cur_csi is not None and cur_csi > self.max_csi:
                    self.save('best')
                    self.best_step = self.cur_step
                    self.max_csi = cur_csi
                self.save('last')
                print_log(f"Valid CSI: {cur_csi}, Best CSI: {self.max_csi} @ step {self.best_step}", self.is_main)
            else:
                self.save()
                print_log("Finish one epoch.", self.is_main)

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    # ============================
    # REQUIRED: test_samples (use your latest Evaluator)
    # ============================
    def test_samples(self, milestone, do_test=False):
        save_vis = True
        data_loader = self.test_loader if do_test else self.valid_loader
        self.model.eval()

        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)

        # === exactly your latest evaluator choice ===
        if do_test:
            from utils.metrics import Evaluator
            evaler = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,   # 固定阈值 [20,30,35,40]
                save_path=save_dir,
            )
        else:
            from utils.metrics_valid import Evaluator
            evaler = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,   # 固定阈值 [20,30,35,40]
                save_path=save_dir,
            )

        cnt = 0
        for batch in tqdm(data_loader, desc='Test Samples', disable=not self.is_main):
            gt, pred = self._sample_batch(batch)  # [B,Tout,1,H,W] torch, [0,1]

            if self.is_main:
                evaler.evaluate(gt, pred)

                if self.args.visual:
                    for i in range(gt.shape[0]):
                        self.visiual_save_fn(
                            pred[i].detach().cpu().numpy(),
                            gt[i].detach().cpu().numpy(),
                            osp.join(save_dir, f"{cnt}-{i}/vil"),
                            data_type='vil'
                        )
                elif save_vis:
                    for i in range(gt.shape[0]):
                        self.visiual_save_fn(
                            pred[i].detach().cpu().numpy(),
                            gt[i].detach().cpu().numpy(),
                            osp.join(save_dir, f"{cnt}-{i}/vil"),
                            data_type='vil'
                        )
                    save_vis = False

            self.accelerator.wait_for_everyone()
            cnt += 1
            if (not do_test) and self.args.valid_limit and cnt >= self.args.vlnum:
                break

        if self.is_main:
            res = evaler.done(is_main_process=self.is_main)
            print_log(f"{'Test' if do_test else 'Valid'} Results: {res}", self.is_main)
            print_log("=" * 30, self.is_main)
            return float(res.get('csi', 0.0))
        return None

    def check_milestones(self, target_ckpt=None):
        mils_paths = os.listdir(self.ckpt_path)
        try:
            milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        except Exception:
            milestones = [m.split('-')[-1].split('.')[0] for m in mils_paths]

        print_log(f"milestones: {milestones}", self.is_main)

        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
            return

        for m in milestones:
            self.load(m)
            self.test_samples(m, do_test=True)


def main():
    args = create_parser()
    if args.gpu_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_use)
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    exp = Runner(args)
    if not args.eval:
        exp.train()
        exp.check_milestones()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)


if __name__ == '__main__':
    main()
