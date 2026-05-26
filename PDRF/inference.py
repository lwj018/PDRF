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

from datasets.get_datasets import get_dataset
from utils.tools import print_log, cycle, show_img_info

#ours
from models.ours.schedulers.rf import RFLOW
from models.ours.unet_wavelet import UKan_Hybrid
rf = RFLOW()

# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "YOUR_WANDB"
os.environ["WANDB_SILENT"] = "true"  # 静默wandb
os.environ["ACCELERATE_DEBUG_MODE"] = "1"


def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',       type=str,   default='ours',        help='backbone model for deterministic prediction')
    parser.add_argument("--seed",           type=int,   default=0,              help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='basic_exps',   help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default=None,           help="additional note for experiment")

    # --------------- Dataset ---------------
    parser.add_argument("--dataset",        type=str,   default='sevir',        help="dataset name")
    parser.add_argument("--img_size",       type=int,   default=128,            help="image size")
    parser.add_argument("--stride",         type=int,   default=13,             help="dataset stride")
    parser.add_argument("--img_channel",    type=int,   default=1,              help="channel of image")
    parser.add_argument("--patch",          type=int,   default=2,              help="patch size")
    parser.add_argument("--seq_len",        type=int,   default=25,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",      type=int,   default=5,              help="number of frames to input")
    parser.add_argument("--frames_out",     type=int,   default=20,             help="number of frames to output")    
    parser.add_argument("--num_workers",    type=int,   default=4,              help="number of workers for data loader")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=1e-4,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=1000,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=8,               help="batch size")
    parser.add_argument("--epochs",         type=int,   default=300,             help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=1,               help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    parser.add_argument("--spec_num",       type=int,   default=20,              help="spectral number")
    parser.add_argument("--layers",         type=int,   default=3,               help="layers number")
    parser.add_argument("--pha_weight",     type=float, default=0.01,            help="phase weight")
    parser.add_argument("--amp_weight",     type=float, default=0.01,            help="amplitute weight")
    parser.add_argument("--anet_weight",    type=float, default=0.1,             help="amplitute network mse weight")
    parser.add_argument("--aw_stop_step",   type=int,   default=5000,            help="training step at which the amplitude weight decays to 0")
    parser.add_argument("--out_weight",     type=float, default=1.0,             help="final output weight")
    parser.add_argument("--tf",             action="store_false",                help="teacher force")
    parser.add_argument("--tf_stop_iter",     type=int,     default=2000,        help="teacher force stop iters")
    parser.add_argument("--tf_changing_rate", type=float,   default=0.,          help="teacher force changing rate")
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    parser.add_argument("--valid",          action="store_true",                 help="valid mode")
    parser.add_argument("--valid_limit",    action="store_true",                 help="valid limit mode")
    parser.add_argument("--vlnum",          type=int,   default=10,              help="valid limit nums")
    parser.add_argument("--visual",         action="store_true",                 help="save all test sample visualization")
    parser.add_argument("--wandb_state",    type=str,   default='disabled',      help="wandb state config")
    parser.add_argument("--gpu_use",        type=str,   nargs='+', default=[],  help="gpu(s) to use")
    parser.add_argument("--res_opt",        action="store_true",                 help="resume opt")

    args = parser.parse_args()
    return args


class InferenceRunner(object):
    
    def __init__(self, args):
        
        self.args = args
        self._preparation()
        
        # Config DDP kwargs from accelerate
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        
        self.accelerator = Accelerator(
            project_config  =   project_config,
            kwargs_handlers =   [ddp_kwargs, process_kwargs],
            mixed_precision =   self.args.mixed_precision,
            log_with        =   None  # 推理时不需要logging
        )
        
        print_log(f"Using GPUs: {self.device}", self.is_main)
        print_log('============================================================', self.is_main)
        print_log("                 Inference Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
        
        self._load_data()
        self.test_loader = self.accelerator.prepare(self.test_loader)
        self._build_model()
        self.model = self.accelerator.prepare(self.model)
        
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)
        else:
            raise ValueError("Please provide checkpoint path for inference")

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.args.backbone
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}_inference"
        
        cur_dir         = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir    = osp.join(cur_dir, 'Inference_Results', self.args.exp_dir, self.exp_name)        
        self.ckpt_path  = osp.join(cur_dir, 'Exps_cikm', self.args.exp_dir, f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}", 'checkpoints')
        self.test_path  = osp.join(self.exp_dir, 'test_samples')
        self.log_path   = osp.join(self.exp_dir, 'logs')
        
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
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
        # =================================
        # Get Test dataloader
        # =================================

        _, _, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
            stride=self.args.stride,
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
        if self.args.dataset != 'sevir':
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
        print_log(f"test_data: {len(self.test_loader)}",
                  self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}",
                  self.is_main)
        
    def _build_model(self):
        # =================================
        # Build model for inference
        # =================================
        print_log("Build Model!", self.is_main)
        
        if self.args.backbone == 'ours':
            model = UKan_Hybrid(T=1000,in_ch=self.args.frames_out,out_ch=self.args.frames_out, ch=64, ch_mult=[1, 2, 2, 2], attn=[],num_res_blocks=2, dropout=0.1).cuda()
        else:
            raise NotImplementedError("Only 'ours' backbone is supported in this inference script")
            
        self.model = model
        print_log("begin ema", self.is_main)
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        print_log("Model built successfully", self.is_main)
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)
        
    def load(self, milestone):
        # =================================
        # load model checkpoint
        # =================================        
        device = self.accelerator.device
        
        if isinstance(milestone, str) and '.pt' in milestone:
            data = torch.load(milestone, map_location=device)
            print_log(f"Load checkpoint {milestone}.", self.is_main)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
            print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)
        self.model = self.accelerator.prepare(model)
        
        if self.is_main:
            self.ema.load_state_dict(data['ema'], strict=False)

    def _get_seq_data(self, batch):
        return batch[:, :self.args.frames_out + self.args.frames_in]       # [B, T, C, H, W]
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False):
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        
        z = torch.randn(radar_input.shape[0],self.args.frames_out,128,128,requires_grad=False).cuda()
        condition=radar_input.squeeze(2)
        pred = rf.sampleno(self.model,z,condition,device = torch.device('cuda:' + str(0)))
        radar_pred=pred.unsqueeze(2)
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()
        radar_input = self.accelerator.gather(radar_input).detach().cpu().numpy()

        return radar_gt, radar_pred, radar_input
    
    def inference(self):
        # =================================
        # Run inference and save results
        # =================================
        
        # 创建结果目录
        save_dir = osp.join(self.test_path, "inference_results")
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        
        # 初始化评估器
        from utils.metrics import Evaluator
        eval = Evaluator(
            seq_len=self.args.frames_out,
            value_scale=self.scale_value,
            thresholds=self.thresholds,
            save_path=save_dir,
        )
        
        # 全局计数器，为每张图片分配唯一ID
        global_img_count = 0
        
        print_log("Starting inference...", self.is_main)
        
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Inference', disable=not self.is_main)):
            # 进行推理
            radar_ori, radar_recon, radar_input = self._sample_batch(batch)
            
            # 评估结果
            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)
                
                # 保存可视化结果
                batch_size = radar_ori.shape[0]
                for i in range(batch_size):
                    # 为每张图片创建独立的文件夹
                    img_dir = osp.join(save_dir, f"sample_{global_img_count:06d}")
                    vil_out_dir = osp.join(img_dir, "vil")
                    vil_in_dir = osp.join(img_dir, "vil_in")
                    
                    os.makedirs(vil_out_dir, exist_ok=True)
                    os.makedirs(vil_in_dir, exist_ok=True)
                    
                    # 保存输出图片 (预测结果 vs 真实值)
                    self.visiual_save_fn(radar_recon[i], radar_ori[i], vil_out_dir, data_type='vil')
                    
                    # 保存输入图片
                    self.visiual_save_fn(radar_input[i], radar_input[i], vil_in_dir, data_type='vil')
                    
                    global_img_count += 1
                    
                    print_log(f"Saved sample {global_img_count-1} to {img_dir}", self.is_main)

            self.accelerator.wait_for_everyone()
        
        # 完成推理，输出结果
        if self.is_main:
            res = eval.done(is_main_process=self.is_main)
            print_log(f"Inference Results: {res}")
            print_log(f"Total samples processed: {global_img_count}")
            print_log(f"Results saved to: {save_dir}")
            print_log("="*50)
            
            # 保存结果摘要
            result_summary = {
                "total_samples": global_img_count,
                "metrics": res,
                "save_directory": save_dir,
                "checkpoint_used": self.args.ckpt_milestone
            }
            
            import json
            with open(osp.join(save_dir, "inference_summary.json"), 'w') as f:
                json.dump(result_summary, f, indent=2)
                
        print_log("Inference completed successfully!", self.is_main)

            
def main():
    args = create_parser()
    
    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # 强制设置为推理模式
    args.eval = True
    args.visual = True
    
    if args.ckpt_milestone is None:
        raise ValueError("Please provide checkpoint path using --ckpt_milestone")
    
    # 创建推理运行器
    inference_runner = InferenceRunner(args)
    
    # 执行推理
    inference_runner.inference()
    

if __name__ == '__main__':
    main()


#python inference.py --eval --visual --ckpt_milestone   resources/ckpt-best.pt