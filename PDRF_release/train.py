"""Train PDRF for radar precipitation nowcasting.

    python train.py --dataset shanghai --frames_in 5 --frames_out 20 \
        --batch_size 8 --epochs 300

Loss (paper Eqs. 7, 13, 14):
    L_total = L_flow + lambda_phy * L_phy
"""
import os
import os.path as osp
import math
import time
import argparse
import logging
import yaml
from tqdm import tqdm
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from datasets.get_datasets import get_dataset
from utils.tools import print_log, cycle
from utils.ema import EMA
from utils.metrics import Evaluator
from pdrf.model import build_pdrf
from flow.crf import CRF
from physics.semi_lagrangian import SemiLagrangianPrior

crf = CRF()

os.environ["WANDB_API_KEY"] = "YOUR_WANDB"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def create_parser():
    parser = argparse.ArgumentParser()

    # --------------- Basic ---------------
    parser.add_argument("--seed", type=int, default=42, help='experiment seed')
    parser.add_argument("--exp_dir", type=str, default='basic_exps', help="experiment directory")
    parser.add_argument("--exp_note", type=str, default=None, help="additional note for experiment")

    # --------------- Dataset ---------------
    parser.add_argument("--dataset", type=str, default='shanghai',
                       choices=['shanghai', 'cikm', 'meteo', 'sevir'], help="dataset name")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--stride", type=int, default=13, help="dataset stride")
    parser.add_argument("--img_channel", type=int, default=1, help="channel of image")
    parser.add_argument("--seq_len", type=int, default=25, help="sequence length sampled from dataset")
    parser.add_argument("--frames_in", type=int, default=5, help="number of frames to input")
    parser.add_argument("--frames_out", type=int, default=20, help="number of frames to output")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")

    # --------------- Optimizer ---------------
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--lr-beta1", type=float, default=0.90, help="beta 1")
    parser.add_argument("--lr-beta2", type=float, default=0.95, help="beta 2")
    parser.add_argument("--l2-norm", type=float, default=0.0, help="weight decay")
    parser.add_argument("--ema_rate", type=float, default=0.95, help="exponential moving average rate")
    parser.add_argument("--scheduler", type=str, default='cosine',
                       choices=['constant', 'linear', 'cosine'], help="lr scheduler")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="warmup steps")
    parser.add_argument("--mixed_precision", type=str, default='no', help="mixed precision training")
    parser.add_argument("--grad_acc_step", type=int, default=1, help="gradient accumulation step")

    # --------------- Training ---------------
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--ckpt_milestone", type=str, default=None, help="resumed checkpoint milestone")
    parser.add_argument("--gpu_use", type=str, nargs='+', default=[], help="gpu(s) to use")
    parser.add_argument("--res_opt", action="store_true", help="resume optimizer")

    # --------------- Validation / ablation ---------------
    parser.add_argument("--eval", action="store_true", help="evaluation mode")
    parser.add_argument("--valid", action="store_true", help="run validation after each epoch")
    parser.add_argument("--valid_limit", action="store_true", help="limit validation batches")
    parser.add_argument("--vlnum", type=int, default=10, help="valid limit nums")
    parser.add_argument("--visual", action="store_true", help="save all test sample visualizations")
    parser.add_argument("--wandb_state", type=str, default='disabled', help="wandb state config")

    return parser.parse_args()


class Runner(object):

    def __init__(self, args):
        self.args = args
        self._preparation()
        self.max_csi, self.best_step = 0.0, 0

        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))

        self.accelerator = Accelerator(
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs, process_kwargs],
            mixed_precision=self.args.mixed_precision,
            log_with='wandb'
        )
        self.extrapolator = SemiLagrangianPrior(
            extrapolation_timestep=self.args.frames_out
        )
        self.accelerator.init_trackers(
            project_name=self.exp_name,
            config=self.args.__dict__,
            init_kwargs={"wandb": {"mode": self.args.wandb_state}}
        )
        print_log(f"Using GPUs: {self.device}", self.is_main)
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

        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")

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
        self.model_name = 'PDRF'
        self.exp_name = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"

        cur_dir = os.path.dirname(os.path.abspath(__file__))

        self.exp_dir = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)
        self.ckpt_path = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path = osp.join(self.exp_dir, 'test_samples')
        self.log_path = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
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
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'train_log.log')),
                logging.StreamHandler()
            ]
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
        self.thresholds = THRESHOLDS
        self.scale_value = PIXEL_SCALE

        if self.args.dataset != 'sevir':
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size * self.args.grad_acc_step,
                shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)

        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, "
                 f"test_data: {len(self.test_loader)}", self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}", self.is_main)

    def _build_model(self):
        print_log("Build PDRF model!", self.is_main)
        model = build_pdrf(frames_in=self.args.frames_in, frames_out=self.args.frames_out).cuda()
        self.model = model

        print_log("begin ema", self.is_main)
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)
        print_log("end device", self.is_main)

        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total / 1e6), self.is_main)

    def _build_optimizer(self):
        num_steps_per_epoch = len(self.train_loader)
        self.global_epochs = self.args.epochs
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch

        self.cur_step, self.cur_epoch = 0, 0
        warmup_steps = self.args.warmup_steps

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps)
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=self.global_steps)
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=self.global_steps)
        else:
            raise ValueError(f"Invalid scheduler: {self.args.scheduler}")

        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = "
                     f"{self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"optimizer: {self.optimizer} with init lr: {self.args.lr}")

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

        if svname is None:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
            print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        else:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{svname}.pt"))
            print_log(f"Save {svname} checkpoint to {self.ckpt_path}", self.is_main)

    def load(self, milestone):
        device = self.accelerator.device

        if isinstance(milestone, str) and '.pt' in milestone:
            data = torch.load(milestone, map_location=device)
            print_log(f"Load checkpoint {milestone}.", self.is_main)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
            print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.model = self.accelerator.prepare(model)
        if self.args.res_opt:
            try:
                self.optimizer.load_state_dict(data['opt'])
                self.scheduler.load_state_dict(data['scheduler'])
            except Exception:
                print_log(f"No optimizer", self.is_main)
            try:
                self.cur_epoch = data['epoch'] + 1
            except Exception:
                print_log(f"No record epoch", self.is_main)
            try:
                self.cur_step = data['step']
            except Exception:
                print_log(f"No record step", self.is_main)

    def train(self):
        pbar = tqdm(
            initial=self.cur_step,
            total=self.global_steps,
            disable=not self.is_main,
        )
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()

            for i, batch in enumerate(self.train_loader):
                with self.accelerator.autocast():
                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])

                    if self.cur_step == 0:
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)

                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                log_dict = {'lr': lr}
                for k, v in loss_dict.items():
                    log_dict[k] = v.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
                pbar.set_description(state_str)

                if i % 20 == 0:
                    logging.info(state_str + '::' + str(log_dict))
                self.ema.update()

                self.cur_step += 1
                pbar.update(1)

                if self.cur_step == 1:
                    """ sanity check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon = self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            if self.is_main:
                                for i in range(radar_ori.shape[0]):
                                    self.visiual_save_fn(radar_recon[i], radar_ori[i],
                                                         osp.join(self.sanity_path, f"{i}/vil"),
                                                         data_type='vil')
                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            if self.args.valid:
                cur_csi = self.test_samples(self.cur_step)
                if self.args.valid_limit:
                    self.save()
                else:
                    if cur_csi is not None and cur_csi > self.max_csi:
                        self.save('best')
                        self.best_step = self.cur_step
                        self.max_csi = cur_csi
                    self.save('last')
                    print_log(f"Valid Results: {cur_csi}, Best csi: {self.max_csi}, "
                             f"Best step: {self.best_step}", self.is_main)
                print_log(f" ========= Finisth one Epoch ==========", self.is_main)
            else:
                self.save()
                print_log(f" ========= Finisth one Epoch ==========", self.is_main)
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def _get_seq_data(self, batch):
        return batch[:, :self.args.frames_out + self.args.frames_in]  # [B, T, C, H, W]

    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in = radar_batch[:, :self.args.frames_in]
        frames_out = radar_batch[:, self.args.frames_in:]
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, \
            "radar sequence length error"

        condition = frames_in.squeeze(2)          # [B, T_in, H, W]
        x_start = frames_out.squeeze(2)           # [B, T_out, H, W]
        # Semi-Lagrangian advection prior X_SL (physics soft teacher)
        x_sl = self.extrapolator.get_extrapolated_condition(
            condition, x_start.shape[1]).detach()

        loss_dict = crf.training_losses(
            model=self.model, x_start=x_start, condition=condition, x_sl=x_sl
        )
        loss = loss_dict["loss"]
        if loss is None:
            raise ValueError("Loss is None, please check the model")
        return {'total_loss': loss,
                'loss_flow': loss_dict['loss_flow'],
                'loss_phy': loss_dict['loss_phy']}

    @torch.no_grad()
    def _sample_batch(self, batch):
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:, :frame_in], radar_batch[:, frame_in:]

        z = torch.randn(radar_input.shape[0], self.args.frames_out,
                        self.args.img_size, self.args.img_size,
                        requires_grad=False).cuda()
        condition = radar_input.squeeze(2)
        pred = crf.sample(self.model, z, condition, device=self.device)
        radar_pred = pred.unsqueeze(2)

        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()

        return radar_gt, radar_pred

    def test_samples(self, milestone, do_test=False):
        save_vis = True
        data_loader = self.test_loader if do_test else self.valid_loader
        self.model.eval()

        cnt = 0
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test \
            else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)

        if do_test:
            eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )
        else:
            from utils.metrics import Evaluator as ValidEvaluator
            eval = ValidEvaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )

        valid_nums = 0
        for batch in tqdm(data_loader, desc='Test Samples', disable=not self.is_main):
            radar_ori, radar_recon = self._sample_batch(batch)
            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)
                if self.args.visual:
                    for i in range(radar_ori.shape[0]):
                        self.visiual_save_fn(radar_recon[i], radar_ori[i],
                                             osp.join(save_dir, f"{cnt}-{i}/vil"), data_type='vil')
                        self.visiual_save_fn(batch[i, :5], batch[i, :5],
                                             osp.join(save_dir, f"{cnt}-{i}/vil_in"), data_type='vil')
                elif save_vis:
                    for i in range(radar_ori.shape[0]):
                        self.visiual_save_fn(radar_recon[i], radar_ori[i],
                                             osp.join(save_dir, f"{cnt}-{i}/vil"), data_type='vil')
                    save_vis = False

            self.accelerator.wait_for_everyone()
            valid_nums += 1
            if not do_test and self.args.valid_limit and valid_nums >= self.args.vlnum:
                break

        if self.is_main:
            res = eval.done(is_main_process=self.is_main)
            if do_test:
                print_log(f"Test Results: {res}")
            else:
                print_log(f"Valid Results: {res}")
            print_log("=" * 30)

            if self.args.valid:
                return res['csi']
        else:
            return None

    def check_milestones(self, target_ckpt=None):
        mils_paths = os.listdir(self.ckpt_path)
        try:
            milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        except Exception:
            milestones = [m.split('-')[-1].split('.')[0] for m in mils_paths]
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)

        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
            return

        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], do_test=True)


def main():
    args = create_parser()
    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    exp = Runner(args)
    if not args.eval:
        exp.train()
        exp.check_milestones()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)


if __name__ == '__main__':
    main()
