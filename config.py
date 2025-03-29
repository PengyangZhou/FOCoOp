"""
this file contains parameters describing the setting (e.g. dataset, backbone model) of the experiment.
variable `parser` is used in `/main.py`.
"""
import argparse

parser = argparse.ArgumentParser(description='arguments for OOD generalization and detection training')

# for prompt_args
parser.add_argument('--ctx_init', default= None) #  "A photo of a {}") # only support false
parser.add_argument("--csc", type=bool, default=False)
parser.add_argument("--num_ctx", type=int, default=16)
parser.add_argument("--class_token_position", type=str, default="end")
parser.add_argument('--num_prompt', type=int, default=1, help="number of prompts")
parser.add_argument('--num_ex_prompt', type=int, default=100, help="number of prompts")
parser.add_argument("--num_shot", type=int, default=8)
# for PromptFolio
parser.add_argument('--frac', type=float, default=0.2, help='the frac for PromptFolio')
# for FedPGP
parser.add_argument('--bottleneck', type=int, default=4, help="number of middle in reparameter")
parser.add_argument('--mu', type=float, default=1, help='The coefficient of contrastive loss')
parser.add_argument('--temp', type=float, default=0.5, help='The tempuature for FePGP')
# for FedOTP
parser.add_argument('--OT', type=str, default='COT', help="type of OT used: Sinkhorn(for standard OT), COT(for unbalanced OT)")
parser.add_argument('--top_percent', type=float, default=1, help='the top_percent of COT, control the mapping size of prompts on the feature map')
parser.add_argument('--eps', type=float, default=0.1, help='the lambada of sinkhorn distance')
parser.add_argument('--thresh', type=float, default=1e-3, help='the thresh of sinkhorn distance')
parser.add_argument('--max_iter', type=int, default=100, help="max iteration of COT")
# for LoCoOp
parser.add_argument("--top_k", type=int, default=5, help='Top k local feature regions') # defaul 200 for ImageNet1K
parser.add_argument('--lambda_local', type=float, default=0.1, help='Coefficient for local feature learning')
parser.add_argument('--precision', default=None) # control using amp precision or not

# for GalLop
parser.add_argument('--num_local_prompt', type=int, default=4, help="number of local prompts")
parser.add_argument('--prompts_batch_size', type=int, default=2, help="number of local prompts")

# for FedLAPT
parser.add_argument('--alpha_mixup', type=float, default=0.1, help='Mixup coefficient alpha for mixup')
parser.add_argument('--beta_mixup', type=float, default=0.1, help='Mixup coefficient beta for mixup')
parser.add_argument('--total_gs_num', type=int, default=100, help="number of samples from gaussian sampling")
parser.add_argument('--queue_capacity', type=int, default=500, help="Size of tensor queue")
parser.add_argument('--iter_recomputation', type=int, default=10, help="Iteration of recomputing queue") # not used note times of calling syn_x, syn_y = self.class_query.sampling_guassian
parser.add_argument('--selected_gs_num', type=int, default=16, help="number of selected samples from gaussian sampling") # not used
parser.add_argument('--gs_loss_weight', type=str, default="1-1-1-1", help='loss weight for gaussian sampling') # not used
parser.add_argument('--loss_weights', type=str, default="0.5-0.5", help='loss weights for LAPT, in the string form \"weight1-weight2-...\"')
parser.add_argument("--soft_split", type=bool, default=True) # only support false
parser.add_argument("--text_center", type=bool, default=False)
parser.add_argument("--use_gs", type=bool, default=False)
parser.add_argument("--pre_queue", type=bool, default=True)
parser.add_argument('--mix_strategy', type=str, default="wccm",choices=['mixup','cutmix', 'manimix','geomix','manimix_wccm','geomix_wccm','mixup_manimix_wccm','mixup_geomix_wccm','wccm'],  help='Using different mixup strategy for augmentation.')
parser.add_argument('--prompttype', type=str, default="unified",choices=['dis_aware','unified', 'class_specific'],  help='Prompt type for CoOp.')
parser.add_argument('--loss_components', type=str, default="multice", choices=['multice','binaryce','entropy','textentropy','textmultice'], help='The loss method.') # question usage

parser.add_argument('--wandb_mode', type=str, choices=['disabled','online', 'offline'], default='online', help='Wandb log mode')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--use_profile', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--method', type=str, default='FOCoOp')
# ---------- dataset partition ----------
parser.add_argument('--id_dataset', type=str, default='cifar100', help='the ID dataset')
parser.add_argument('--leave_out', type=str, default='art_painting')
parser.add_argument('--dataset_path', type=str, default='/path/to/datasets', help='path to dataset')
parser.add_argument('--alpha', type=float, default=0.1, help='parameter of dirichlet distribution')
parser.add_argument('--num_client', type=int, default=10, help='number of clients')
parser.add_argument('--dataset_seed', type=int, default=21, help='seed to split dataset')
parser.add_argument('--pathological', action="store_true", help='using pathological method split dataset')
parser.add_argument('--non_overlap', action="store_true", help='using iid method split dataset')
parser.add_argument('--class_per_client', type=int, default=20, help='classes per client')
parser.add_argument('--num_classes', type=int, default=100, help='number of dataset classes')
# ---------- backbone ----------
parser.add_argument('--backbone', type=str, choices=['resnet', 'clip'], default='clip', help='backbone model of task')
# ---------- device ----------
parser.add_argument('--device', type=str, default='cuda:0', help='device')
# ---------- server configuration ----------
parser.add_argument('--join_ratio', type=float, default=1., help='join ratio')
parser.add_argument('--communication_rounds', type=int, default=10, help='total communication round')
parser.add_argument('--checkpoint_path', type=str, default='default', help='check point path')
# ---------- client configuration ----------
parser.add_argument('--local_epochs', type=int, default=2, help='local epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') #128
parser.add_argument('--num_workers', type=int, default=2, help='dataloader\'s num_workers')
parser.add_argument('--pin_memory', type=bool, default=False, help='dataloader\'s pin_memory')
# ---------- optimizer --------
parser.add_argument('--learning_rate', type=float, default= 0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)


parser.add_argument("--ema", type=float, default=0.5)
parser.add_argument("--gamma_1", type=float, default=10)
parser.add_argument("--gamma_2", type=float, default=10)
parser.add_argument("--iter_id", type=int, default=5)
parser.add_argument("--iter_ood", type=int, default=5)
parser.add_argument("--noise_strength", type=int, default=1e-3)
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument("--dro", type=bool, default=False)
parser.add_argument("--uot", type=bool, default=False)

# TESTING
parser.add_argument("--score_method", type=str, choices=['msp', 'energy','neglabel'], default='msp')
