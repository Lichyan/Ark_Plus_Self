import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *

import torch
from engine import classification_engine

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = OptionParser()

    parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                      callback=vararg_callback_int)
    parser.add_option("--model", dest="model_name", help="vit_base|vit_small|swin_base|swin_tiny", default="vit_base", type="string")
    parser.add_option("--init", dest="init",
                      help="Random| ImageNet_1k| ImageNet_21k| SAM| DeiT| BEiT| DINO| MoCo_V3| MoBY | MAE| SimMIM",
                      default="Random", type="string")
    parser.add_option("--pretrained_weights", dest="pretrained_weights", help="Path to the Pretrained model", default=None, type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                      default=14, type="int")
    parser.add_option("--num_class_grade", dest="num_class_grade", help="number of grade ordinal logits", default=3, type="int")
    parser.add_option("--num_class_stage", dest="num_class_stage", help="number of stage ordinal logits", default=2, type="int")
    parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert|Shenzhen|VinDrCXR|RSNAPneumonia|advCheX|advCheX_binary|advCheX_hyp|advCheX_hyp_multi_level|advCheX_hyp_multi_stage_v1|advCheX_hyp_multi_stage_v2|advCheX_hyp_multi_grade_stage_v1|advCheX_hyp_multi_grade_stage_sep_v1|advCheX_hyp_grade_stage_v2|advCheX_hyp_grade_stage_embtab_base|advCheX_hyp_grade_stage_embtab_v2lite", default="ChestXray14", type="string")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--img_size", dest="img_size", help="resize image resolution", default=256, type="int")
    parser.add_option("--input_size", dest="input_size", help="input image resolution", default=224, type="int")
    parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default=None, type="string")
    parser.add_option("--train_list", dest="train_list", help="file for training list",
                      default=None, type="string")
    parser.add_option("--val_list", dest="val_list", help="file for validating list",
                      default=None, type="string")
    parser.add_option("--val_data_dir", dest="val_data_dir", help="optional root dir for validation images",
                      default=None, type="string")
    parser.add_option("--test_list", dest="test_list", help="file for test list",
                      default=None, type="string")
    parser.add_option("--train_weights", dest="train_weights", help="optional sampling weight file for training", default=None, type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=64, type="int")
    parser.add_option("--epochs", dest="epochs", help="num of epoches", default=200, type="int")
    parser.add_option("--exp_name", dest="exp_name", default="", type="string")
    parser.add_option("--key", help="key name in the pretrained checkpoint", default="state_dict")
    parser.add_option("--freeze_encoder", dest="freeze_encoder", help="whether freeze encoder", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--use_lora", dest="use_lora", help="whether enable LoRA adapters", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--lora_rank", dest="lora_rank", help="LoRA rank", default=8, type="int")
    parser.add_option("--lora_alpha", dest="lora_alpha", help="LoRA alpha scaling", default=16.0, type="float")
    parser.add_option("--lora_dropout", dest="lora_dropout", help="LoRA dropout", default=0.0, type="float")
    parser.add_option("--lora_targets", dest="lora_targets",
                      help="comma separated module name patterns to inject LoRA",
                      default="attn.qkv,attn.proj,mlp.fc1,mlp.fc2", type="string")
    parser.add_option("--lora_train_head", dest="lora_train_head",
                      help="keep classification head trainable when using LoRA", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--skip_training", dest="skip_training", help="whether skip training", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--test_every_epoch", dest="test_every_epoch", help="whether skip training", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--scale_up", dest="scale_up", help="whether scale up resolution", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--keep_head", dest="keep_head", help="retain head.* weights when loading checkpoint", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--skip_test", dest="skip_test", help="skip evaluation after training", default=False,
                      action="callback", callback=vararg_callback_bool)
    # Optimizer parameters
    parser.add_option('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_option('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_option('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_option('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_option('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_option('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_option('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_option('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_option('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_option('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_option('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_option('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_option('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_option('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_option('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_option('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_option('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')


    parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
    parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
    parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
    parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
    parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                      default=True, action="callback", callback=vararg_callback_bool)
    parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")
    parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")
    parser.add_option("--uncertain_label", dest="uncertain_label",
                      help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
                      default="LSR-Ones", type="string")
    parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
                      default=0, type="int")
    parser.add_option("--weighted_BCELoss", dest="weighted_BCELoss", help="whether use weighted BCELoss", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--loss_fn", dest="loss_fn", help="loss function: bce | focal", default="bce", type="string")
    parser.add_option("--focal_alpha", dest="focal_alpha", help="alpha for focal loss", default=0.25, type="float")
    parser.add_option("--focal_gamma", dest="focal_gamma", help="gamma for focal loss", default=2.0, type="float")
    parser.add_option("--pos_weight", dest="pos_weight", help="comma separated pos_weight for ordinal tasks (len=3)", default=None, type="string")
    parser.add_option("--pos_weight_grade", dest="pos_weight_grade", help="comma separated pos_weight for grade head (len=3)", default=None, type="string")
    parser.add_option("--pos_weight_stage", dest="pos_weight_stage", help="comma separated pos_weight for stage head (len=2)", default=None, type="string")
    parser.add_option(
        "--ordinal_mode",
        dest="ordinal_mode",
        help="ordinal mode for multi-head grade/stage: coral|corn",
        default="coral",
        type="string",
    )
    parser.add_option("--loss_w_grade", dest="loss_w_grade", help="grade head loss weight", default=1.0, type="float")
    parser.add_option("--loss_w_stage", dest="loss_w_stage", help="stage head loss weight", default=1.0, type="float")
    parser.add_option("--use_joint_train", dest="use_joint_train", help="whether use joint training loss", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--lambda_incomp", dest="lambda_incomp", help="weight for incompatibility loss", default=0.0, type="float")
    parser.add_option("--lambda_joint", dest="lambda_joint", help="weight for joint loss", default=0.0, type="float")
    parser.add_option("--joint_gate", dest="joint_gate", help="none|htn_only", default="htn_only", type="string")
    parser.add_option("--joint_detach", dest="joint_detach", help="none|grade|stage|both", default="both", type="string")
    parser.add_option("--joint_ce_weight_mode", dest="joint_ce_weight_mode", help="none|inv|inv_sqrt", default="inv_sqrt", type="string")
    parser.add_option("--joint_warmup_epochs", dest="joint_warmup_epochs", help="warmup epochs for joint/incomp", default=5, type="int")
    parser.add_option("--incomp_mode", dest="incomp_mode", help="mask_sum|log_barrier", default="mask_sum", type="string")
    parser.add_option("--joint_loss_use_prior", dest="joint_loss_use_prior", help="use prior in joint CE loss", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--joint_prior_mode", dest="joint_prior_mode", help="none|mimic|mix", default="mimic", type="string")
    parser.add_option("--joint_prior_alpha", dest="joint_prior_alpha", help="alpha for joint prior", default=0.2, type="float")
    parser.add_option("--joint_prior_eps", dest="joint_prior_eps", help="eps for joint prior smoothing", default=1e-3, type="float")
    parser.add_option("--joint_prior_beta", dest="joint_prior_beta", help="beta for mix prior", default=0.5, type="float")
    parser.add_option("--joint_prior_private_json", dest="joint_prior_private_json", help="private prior json for mix", default=None, type="string")
    parser.add_option("--softacc_gamma_over", dest="softacc_gamma_over", help="gamma for over-triage in soft acc", default=0.5, type="float")
    parser.add_option("--modethese", dest="modethese", help="enable extended metrics/figures for论文需求", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--thresholds_json", dest="thresholds_json", help="optional json file that stores thresholds for ordinal eval", default=None, type="string")
    parser.add_option("--decodermode", dest="decodermode", help="non|threshold|ev|temp_threshold|temp_ev", default="non", type="string")
    parser.add_option("--decoder_objective", dest="decoder_objective", help="qwk|macro_f1|balanced_acc|mid_recall|composite", default="qwk", type="string")
    parser.add_option("--decoder_bins", dest="decoder_bins", help="grid bins for decoder search", default=101, type="int")
    parser.add_option("--decoder_use_saved_thresholds", dest="decoder_use_saved_thresholds", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--decoder_save_debug", dest="decoder_save_debug", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--temperature_init", dest="temperature_init", default=1.0, type="float")
    parser.add_option("--temperature_min", dest="temperature_min", default=0.5, type="float")
    parser.add_option("--temperature_max", dest="temperature_max", default=5.0, type="float")
    parser.add_option("--temperature_grid_size", dest="temperature_grid_size", default=91, type="int")
    parser.add_option("--decoder_keep_raw_metrics", dest="decoder_keep_raw_metrics", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--sep_head_mode", dest="sep_head_mode", help="flat|coarse_fine", default="flat", type="string")
    parser.add_option("--loss_w_anyhtn", dest="loss_w_anyhtn", help="coarse any-HTN loss weight", default=1.0, type="float")
    parser.add_option("--pos_weight_anyhtn", dest="pos_weight_anyhtn", help="optional positive weight for any-HTN coarse head", default=None, type="string")
    parser.add_option("--coarse_auc_loss_mode", dest="coarse_auc_loss_mode", help="none|pairwise_hinge|pairwise_logistic", default="none", type="string")
    parser.add_option("--loss_w_anyhtn_auc", dest="loss_w_anyhtn_auc", help="coarse AUC-oriented loss alpha", default=0.0, type="float")
    parser.add_option("--auc_margin", dest="auc_margin", help="margin for pairwise hinge AUC loss", default=1.0, type="float")
    parser.add_option("--auc_pair_subsample", dest="auc_pair_subsample", help="max sampled positives/negatives per batch for pairwise AUC", default=256, type="int")
    parser.add_option("--auc_loss_detach_probs", dest="auc_loss_detach_probs", help="detach coarse logits before AUC loss", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--fine_soft_label_mode", dest="fine_soft_label_mode", help="none|grade_only|grade_and_stage", default="none", type="string")
    parser.add_option("--grade_soft_center", dest="grade_soft_center", help="center mass for positive-grade soft labels", default=0.85, type="float")
    parser.add_option("--stage_label_smoothing", dest="stage_label_smoothing", help="label smoothing epsilon for positive-stage head", default=0.05, type="float")
    parser.add_option("--loss_w_grade_soft", dest="loss_w_grade_soft", help="aux soft-label loss weight for positive-grade head", default=0.2, type="float")
    parser.add_option("--loss_w_stage_soft", dest="loss_w_stage_soft", help="aux soft-label loss weight for full-stage head in v1", default=0.1, type="float")
    parser.add_option("--loss_w_stage_smooth", dest="loss_w_stage_smooth", help="optional scale for stage smoothing BCE", default=1.0, type="float")
    parser.add_option("--img_emb_dim", dest="img_emb_dim", default=1376, type="int")
    parser.add_option("--tab_dim", dest="tab_dim", default=5, type="int")
    parser.add_option("--img_hidden_dim", dest="img_hidden_dim", default=512, type="int")
    parser.add_option("--img_out_dim", dest="img_out_dim", default=256, type="int")
    parser.add_option("--tab_hidden_dim", dest="tab_hidden_dim", default=32, type="int")
    parser.add_option("--tab_out_dim", dest="tab_out_dim", default=64, type="int")
    parser.add_option("--fusion_hidden_dim", dest="fusion_hidden_dim", default=192, type="int")
    parser.add_option("--task_hidden_dim", dest="task_hidden_dim", default=128, type="int")
    parser.add_option("--dropout_img", dest="dropout_img", default=0.2, type="float")
    parser.add_option("--dropout_tab", dest="dropout_tab", default=0.1, type="float")
    parser.add_option("--dropout_fusion", dest="dropout_fusion", default=0.2, type="float")
    parser.add_option("--grade_tab_scale", dest="grade_tab_scale", default=0.3, type="float")
    parser.add_option("--v1_soft_label_mode", dest="v1_soft_label_mode", help="none|full (apply full-grade/full-stage soft distribution loss for v1)", default="none", type="string")
    parser.add_option("--grade_soft_scheme", dest="grade_soft_scheme", help="soft target scheme for v1 grade full distribution", default="asym_v1", type="string")
    parser.add_option("--stage_soft_scheme", dest="stage_soft_scheme", help="soft target scheme for v1 stage full distribution", default="asym_v1", type="string")
    parser.add_option("--lambda_stage_marg", dest="lambda_stage_marg", default=0.8, type="float")
    parser.add_option("--lambda_cond_stage", dest="lambda_cond_stage", default=0.6, type="float")
    parser.add_option("--lambda_soft_joint", dest="lambda_soft_joint", default=0.15, type="float")
    parser.add_option("--lambda_cond", dest="lambda_cond", default=0.5, type="float")
    parser.add_option("--lambda_joint_soft", dest="lambda_joint_soft", default=0.05, type="float")
    parser.add_option("--stage_fused_aux_weight", dest="stage_fused_aux_weight", default=0.3, type="float")
    parser.add_option("--cond_pos_weight_g1", dest="cond_pos_weight_g1", default=3.0, type="float")
    parser.add_option("--cond_pos_weight_g2", dest="cond_pos_weight_g2", default=5.0, type="float")
    parser.add_option("--joint_graph_tau", dest="joint_graph_tau", default=0.7, type="float")
    parser.add_option("--joint_beta_stage", dest="joint_beta_stage", default=0.5, type="float")
    parser.add_option("--joint_gamma_cond", dest="joint_gamma_cond", default=0.5, type="float")
    parser.add_option("--use_stopgrad_grade_for_cond", dest="use_stopgrad_grade_for_cond", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--use_residual_gated_fusion", dest="use_residual_gated_fusion", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--gate_hidden_dim", dest="gate_hidden_dim", default=128, type="int")
    parser.add_option("--use_v2lite_fused_eval", dest="use_v2lite_fused_eval", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--use_legal_joint_composer", dest="use_legal_joint_composer", default=True,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--v2_soft_joint_start_epoch", dest="v2_soft_joint_start_epoch", default=5, type="int")
    parser.add_option("--v2_soft_joint_warmup_epochs", dest="v2_soft_joint_warmup_epochs", default=5, type="int")
    parser.add_option("--teacher_force_grade_epochs", dest="teacher_force_grade_epochs", default=0, type="int")
    parser.add_option("--alpha_gate_min", dest="alpha_gate_min", default=0.15, type="float")
    parser.add_option("--alpha_gate_max", dest="alpha_gate_max", default=0.65, type="float")
    parser.add_option("--lpv3_enable_neck", dest="lpv3_enable_neck", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--lpv3_neck_hidden_dim", dest="lpv3_neck_hidden_dim", default=512, type="int")
    parser.add_option("--lpv3_neck_out_dim", dest="lpv3_neck_out_dim", default=128, type="int")
    parser.add_option("--lpv3_neck_dropout", dest="lpv3_neck_dropout", default=0.2, type="float")
    parser.add_option("--lpv3_joint_aware_sampler", dest="lpv3_joint_aware_sampler", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--lpv3_sampler_mode", dest="lpv3_sampler_mode", default="joint_inv_freq", type="string")
    parser.add_option("--lpv3_sampler_power", dest="lpv3_sampler_power", default=0.5, type="float")
    parser.add_option("--lpv3_sampler_cap", dest="lpv3_sampler_cap", default=5.0, type="float")
    parser.add_option("--lpv3_sampler_floor", dest="lpv3_sampler_floor", default=1.0, type="float")
    parser.add_option("--lpv3_sampler_boost_11", dest="lpv3_sampler_boost_11", default=2.0, type="float")
    parser.add_option("--lpv3_sampler_boost_21", dest="lpv3_sampler_boost_21", default=4.0, type="float")
    parser.add_option("--lpv3_sampler_boost_32", dest="lpv3_sampler_boost_32", default=1.5, type="float")
    parser.add_option("--lpv3_sampler_boost_12", dest="lpv3_sampler_boost_12", default=1.0, type="float")
    parser.add_option("--lpv3_sampler_boost_22", dest="lpv3_sampler_boost_22", default=1.0, type="float")
    parser.add_option("--lpv3_stageA_epochs", dest="lpv3_stageA_epochs", default=5, type="int")
    parser.add_option("--lpv3_enable_cond_after_epoch", dest="lpv3_enable_cond_after_epoch", default=3, type="int")
    parser.add_option("--lpv3_enable_soft_joint_after_epoch", dest="lpv3_enable_soft_joint_after_epoch", default=10, type="int")
    parser.add_option("--joint_graph_w_00_11", dest="joint_graph_w_00_11", default=1.0, type="float")
    parser.add_option("--joint_graph_w_11_21", dest="joint_graph_w_11_21", default=0.6, type="float")
    parser.add_option("--joint_graph_w_11_12", dest="joint_graph_w_11_12", default=1.2, type="float")
    parser.add_option("--joint_graph_w_21_22", dest="joint_graph_w_21_22", default=0.8, type="float")
    parser.add_option("--joint_graph_w_12_22", dest="joint_graph_w_12_22", default=0.7, type="float")
    parser.add_option("--joint_graph_w_22_32", dest="joint_graph_w_22_32", default=1.5, type="float")
    parser.add_option("--test_time_adjust", dest="test_time_adjust", help="在测试集上重新寻阈值", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option("--output_special", dest="output_special", help="输出TP/FP/TN/FN样本示例", default=False,
                      action="callback", callback=vararg_callback_bool)
    parser.add_option('--few_shot', dest="few_shot", help='number or percentage of training samples', default=-1, type=float)


    (options, args) = parser.parse_args()

    return options


def main(args):
    print(args)
    assert args.data_dir is not None
    # assert args.train_list is not None
    # assert args.val_list is not None
    # assert args.test_list is not None
    #if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
    #    assert args.proxy_dir is not None
    args.exp_name = args.model_name + "_" + args.init + args.exp_name
    model_path = os.path.join("./Models/Classification",args.data_set)
    output_path = os.path.join("./Outputs/Classification",args.data_set)

    if args.data_set == "ChestXray14":
        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        dataset_train = ChestXray14(images_path=args.data_dir, file_path=args.train_list,
                                           augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size),few_shot=args.few_shot)

        dataset_val = ChestXray14(images_path=args.data_dir, file_path=args.val_list,
                                         augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))
        dataset_test = ChestXray14(images_path=args.data_dir, file_path=args.test_list,
                                          augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)


    elif args.data_set == "CheXpert":
        diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                           'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        test_diseases_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        test_diseases = [diseases.index(c) for c in test_diseases_name]
        dataset_train = CheXpert(images_path=args.data_dir, file_path=args.train_list,
                                        augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, few_shot=args.few_shot)

        dataset_val = CheXpert(images_path=args.data_dir, file_path=args.val_list,
                                      augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)

        dataset_test = CheXpert(images_path=args.data_dir, file_path=args.test_list,
                                       augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases)

    elif args.data_set == "Shenzhen":
        diseases = ['TB']
        dataset_train = ShenzhenCXR(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), few_shot=args.few_shot)

        dataset_val = ShenzhenCXR(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))

        dataset_test = ShenzhenCXR(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    
    elif args.data_set == "VinDrCXR":
        diseases = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        dataset_train = VinDrCXR(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), few_shot=args.few_shot)

        dataset_val = VinDrCXR(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))

        dataset_test = VinDrCXR(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    elif args.data_set == "VinDrCXR_all":
        diseases = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        dataset_train = VinDrCXR_all(images_path=args.data_dir, file_path=args.train_list,diseases=diseases,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), few_shot=args.few_shot)

        dataset_val = VinDrCXR_all(images_path=args.data_dir, file_path=args.val_list,diseases=diseases,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))

        dataset_test = VinDrCXR_all(images_path=args.data_dir, file_path=args.test_list,diseases=diseases,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "RSNAPneumonia":
        diseases = ['Normal', 'No Lung Opacity/Not Normal', 'Lung Opacity']
        dataset_train = RSNAPneumonia(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), few_shot=args.few_shot)

        dataset_val = RSNAPneumonia(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))

        dataset_test = RSNAPneumonia(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)                           
    elif args.data_set == "COVIDx":
        diseases = ['normal', 'pneumonia', 'COVID-19']
        dataset_train = COVIDx(images_path=os.path.join(args.data_dir, 'train'), file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), classes = diseases, few_shot=args.few_shot)

        dataset_val = COVIDx(images_path=os.path.join(args.data_dir, 'test'), file_path=args.val_list, classes = diseases,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))

        dataset_test = COVIDx(images_path=os.path.join(args.data_dir, 'test'), file_path=args.test_list, classes = diseases,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))

        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
  
    elif args.data_set == "MIMIC":
        diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                           'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        dataset_train = MIMIC(images_path=args.data_dir, file_path=args.train_list,
                                        augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, few_shot=args.few_shot)

        dataset_val = MIMIC(images_path=args.data_dir, file_path=args.val_list,
                                      augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)

        dataset_test = MIMIC(images_path=args.data_dir, file_path=args.test_list,
                                       augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size), uncertain_label="Ones", unknown_label=args.unknown_label)

        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
                    
    elif args.data_set == "ChestDR":
        diseases = ['pleural_effusion','nodule','pneumonia','cardiomegaly','hilar_enlargement',
                    'fracture_old','fibrosis','aortic_calcification','tortuous_aorta',
                    'thickened_pleura','TB','pneumothorax','emphysema','atelectasis',
                    'calcification','pulmonary_edema','increased_lung_markings',
                    'elevated_diaphragm','consolidation']
        dataset_train = ChestDR(images_path=args.data_dir, file_path=args.train_list,
                                        augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize = args.img_size), few_shot= args.few_shot)

        dataset_val = ChestDR(images_path=args.data_dir, file_path=args.val_list,
                                      augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize = args.img_size))

        dataset_test = ChestDR(images_path=args.data_dir, file_path=args.test_list,
                                       augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize = args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    
    elif args.data_set == "advCheX_old":
        # 定义19类疾病名称（与你的CSV表头一致）
        diseases = ['Normal', 'ASD', 'VSD', 'PDA', 'TOF', 'MS', 'PS', 'AS', 
                    'AR', 'MR', 'PAH', 'PFO', 'HCM', 'DCM', 'ARVC', 'CAD', 
                    'HTN', 'Aneurysm', 'Other']
        if args.mode == "train":
        # 创建训练集
            dataset_train = advCheX(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization, 
                    mode="train", 
                    crop_size=args.input_size, 
                    resize=args.img_size
                ),
                num_class=19,
                few_shot=args.few_shot
            )
            
            # 创建验证集
            dataset_val = advCheX(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization, 
                    mode="valid", 
                    crop_size=args.input_size, 
                    resize=args.img_size
                ),
                num_class=19
            )
        else:
            dataset_train = None 
            dataset_val = None
        # 创建测试集
        dataset_test = advCheX(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization, 
                mode="test", 
                crop_size=args.input_size, 
                resize=args.img_size
            ),
            num_class=19
        )
        def _summ(ds, name):
            y = np.array(ds.img_label)
            print(f"[{name}] shape={y.shape}, positives_per_class={y.sum(axis=0).tolist()}")
            # 检查每行列数
            bad_rows = [i for i, row in enumerate(ds.img_label) if len(row) != 19]
            print(f"[{name}] rows_with_len!=19: {len(bad_rows)}")
        if args.mode == "train":    
            _summ(dataset_train, "train")
            _summ(dataset_val, "val")
        _summ(dataset_test,  "test")
        
        # 启动分类训练引擎
        classification_engine(args, model_path, output_path, diseases, 
                             dataset_train, dataset_val, dataset_test)
        
    elif args.data_set == "advCheX":
        # 定义3类疾病名称（与你的CSV表头一致）
        diseases = ['CHD', 'nonCHD', 'Other']
        if args.mode == "train":
        # 创建训练集
            dataset_train = advCheX(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization, 
                    mode="train", 
                    crop_size=args.input_size, 
                    resize=args.img_size
                ),
                num_class=3,
                few_shot=args.few_shot
            )
            
            # 创建验证集
            dataset_val = advCheX(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization, 
                    mode="valid", 
                    crop_size=args.input_size, 
                    resize=args.img_size
                ),
                num_class=3
            )
        else:
            dataset_train = None 
            dataset_val = None
        # 创建测试集
        dataset_test = advCheX(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization, 
                mode="test", 
                crop_size=args.input_size, 
                resize=args.img_size
            ),
            num_class=3
        )
        def _summ(ds, name):
            y = np.array(ds.img_label)
            print(f"[{name}] shape={y.shape}, positives_per_class={y.sum(axis=0).tolist()}")
            # 检查每行列数
            bad_rows = [i for i, row in enumerate(ds.img_label) if len(row) != 3]
            print(f"[{name}] rows_with_len!=3: {len(bad_rows)}")
        if args.mode == "train":    
            _summ(dataset_train, "train")
            _summ(dataset_val, "val")
        _summ(dataset_test,  "test")
        
        # 启动分类训练引擎
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_binary":
        diseases = ['CHD', 'nonCHD']
        if args.mode == "train":
            dataset_train = advCheX_binary(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                num_class=2,
                few_shot=args.few_shot
            )
            dataset_val = advCheX_binary(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                num_class=2
            )
        else:
            dataset_train = None
            dataset_val = None
        dataset_test = advCheX_binary(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
            num_class=2
        )
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp":
        label_names = None
        if args.mode == "train":
            dataset_train = advCheX_hyp(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                num_class=2,
                few_shot=args.few_shot
            )
            label_names = getattr(dataset_train, "label_names", None)
            dataset_val = advCheX_hyp(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                num_class=2
            )
        else:
            dataset_train = None
            dataset_val = None
        dataset_test = advCheX_hyp(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
            num_class=2
        )
        if not label_names:
            label_names = getattr(dataset_test, "label_names", None)
        diseases = label_names if label_names else ['Hypertension', 'nonHypertension']
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_multi_level":
        # ordinal 0~3 -> 3 thresholds (>=1, >=2, >=3)
        args.num_class = 3
        label_names = [">=1", ">=2", ">=3"]
        if args.mode == "train":
            dataset_train = advCheX_hyp_multi_level(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                few_shot=args.few_shot,
            )
            dataset_val = advCheX_hyp_multi_level(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
            )
        else:
            dataset_train = None
            dataset_val = None

        dataset_test = advCheX_hyp_multi_level(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
        )
        diseases = label_names
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_multi_grade_stage_v1":
        args.num_class_grade = 3
        args.num_class_stage = 2
        args.num_class = args.num_class_grade
        label_names = ["grade>=1", "grade>=2", "grade>=3", "stage>=1", "stage>=2"]
        need_decoder_val = str(getattr(args, "decodermode", "non")).lower() != "non"
        if args.mode == "train":
            dataset_train = advCheX_hyp_multi_grade_stage_v1(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                few_shot=args.few_shot,
            )
            val_images_root = args.val_data_dir if getattr(args, "val_data_dir", None) else args.data_dir
            dataset_val = advCheX_hyp_multi_grade_stage_v1(
                images_path=val_images_root,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
            )
        else:
            dataset_train = None
            if need_decoder_val:
                val_images_root = args.val_data_dir if getattr(args, "val_data_dir", None) else args.data_dir
                print(f"[v1] val_images_root={val_images_root}, val_list={args.val_list}", flush=True)
                dataset_val = advCheX_hyp_multi_grade_stage_v1(
                    images_path=val_images_root,
                    file_path=args.val_list,
                    augment=build_transform_classification(
                        normalize=args.normalization,
                        mode="valid",
                        crop_size=args.input_size,
                        resize=args.img_size
                    ),
                )
            else:
                dataset_val = None

        dataset_test = advCheX_hyp_multi_grade_stage_v1(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
        )
        diseases = label_names
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_grade_stage_v2":
        diseases = ['grade', 'stage']
        val_images_root = args.val_data_dir if getattr(args, "val_data_dir", None) else args.data_dir
        if args.mode == "train":
            dataset_train = advCheX_hyp_grade_stage_v2(images_path=args.data_dir, file_path=args.train_list,
                augment=build_transform_classification(normalize=args.normalization, mode="train", crop_size=args.input_size, resize=args.img_size),
                few_shot=args.few_shot)
            dataset_val = advCheX_hyp_grade_stage_v2(images_path=val_images_root, file_path=args.val_list,
                augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize=args.img_size))
        else:
            dataset_train = None
            dataset_val = advCheX_hyp_grade_stage_v2(images_path=val_images_root, file_path=args.val_list,
                augment=build_transform_classification(normalize=args.normalization, mode="valid", crop_size=args.input_size, resize=args.img_size)) if args.val_list else None
        dataset_test = advCheX_hyp_grade_stage_v2(images_path=args.data_dir, file_path=args.test_list,
            augment=build_transform_classification(normalize=args.normalization, mode="test", crop_size=args.input_size, resize=args.img_size))
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_grade_stage_embtab_base":
        args.num_class_grade = 3
        args.num_class_stage = 2
        args.num_class = args.num_class_grade
        diseases = ["grade", "stage"]
        val_images_root = args.val_data_dir if getattr(args, "val_data_dir", None) else args.data_dir
        need_decoder_val = str(getattr(args, "decodermode", "non")).lower() != "non"
        if args.mode == "train":
            dataset_train = advCheX_hyp_grade_stage_embtab_base(
                images_path=args.data_dir,
                file_path=args.train_list,
                split="train",
                tab_norm_stats=None,
                few_shot=args.few_shot,
            )
            dataset_val = advCheX_hyp_grade_stage_embtab_base(
                images_path=val_images_root,
                file_path=args.val_list,
                split="valid",
                tab_norm_stats=dataset_train.tab_norm_stats,
            )
        else:
            dataset_train = None
            if need_decoder_val and args.val_list:
                dataset_val_train = advCheX_hyp_grade_stage_embtab_base(
                    images_path=args.data_dir,
                    file_path=args.train_list,
                    split="train",
                    tab_norm_stats=None,
                )
                dataset_val = advCheX_hyp_grade_stage_embtab_base(
                    images_path=val_images_root,
                    file_path=args.val_list,
                    split="valid",
                    tab_norm_stats=dataset_val_train.tab_norm_stats,
                )
            else:
                dataset_val = None

        tab_stats_for_test = None
        if args.mode == "train" and dataset_train is not None:
            tab_stats_for_test = dataset_train.tab_norm_stats
        elif args.train_list:
            dataset_train_for_stats = advCheX_hyp_grade_stage_embtab_base(
                images_path=args.data_dir,
                file_path=args.train_list,
                split="train",
                tab_norm_stats=None,
            )
            tab_stats_for_test = dataset_train_for_stats.tab_norm_stats
        dataset_test = advCheX_hyp_grade_stage_embtab_base(
            images_path=args.data_dir,
            file_path=args.test_list,
            split="test",
            tab_norm_stats=tab_stats_for_test,
        )
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite":
        args.num_class_grade = 3
        args.num_class_stage = 2
        args.num_class = args.num_class_grade
        diseases = ["grade", "stage"]
        val_images_root = args.val_data_dir if getattr(args, "val_data_dir", None) else args.data_dir
        need_decoder_val = str(getattr(args, "decodermode", "non")).lower() != "non"
        if args.mode == "train":
            dataset_train = advCheX_hyp_grade_stage_embtab_v2lite(
                images_path=args.data_dir, file_path=args.train_list, split="train", tab_norm_stats=None, few_shot=args.few_shot
            )
            dataset_val = advCheX_hyp_grade_stage_embtab_v2lite(
                images_path=val_images_root, file_path=args.val_list, split="valid", tab_norm_stats=dataset_train.tab_norm_stats
            )
        else:
            dataset_train = None
            if need_decoder_val and args.val_list:
                dataset_val_train = advCheX_hyp_grade_stage_embtab_v2lite(
                    images_path=args.data_dir, file_path=args.train_list, split="train", tab_norm_stats=None
                )
                dataset_val = advCheX_hyp_grade_stage_embtab_v2lite(
                    images_path=val_images_root, file_path=args.val_list, split="valid", tab_norm_stats=dataset_val_train.tab_norm_stats
                )
            else:
                dataset_val = None
        tab_stats_for_test = None
        if args.mode == "train" and dataset_train is not None:
            tab_stats_for_test = dataset_train.tab_norm_stats
        elif args.train_list:
            dataset_train_for_stats = advCheX_hyp_grade_stage_embtab_v2lite(
                images_path=args.data_dir, file_path=args.train_list, split="train", tab_norm_stats=None
            )
            tab_stats_for_test = dataset_train_for_stats.tab_norm_stats
        dataset_test = advCheX_hyp_grade_stage_embtab_v2lite(
            images_path=args.data_dir, file_path=args.test_list, split="test", tab_norm_stats=tab_stats_for_test
        )
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_multi_grade_stage_sep_v1":
        args.num_class_grade = 3
        args.num_class_stage = 2
        args.num_class = args.num_class_grade
        label_names = ["grade>=1", "grade>=2", "grade>=3", "stage>=1", "stage>=2"]
        need_decoder_val = str(getattr(args, "decodermode", "non")).lower() != "non"
        if args.mode == "train":
            dataset_train = advCheX_hyp_multi_grade_stage_sep_v1(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                few_shot=args.few_shot,
            )
        else:
            dataset_train = None

        val_images_root = args.val_data_dir if getattr(args, "val_data_dir", None) else args.data_dir
        if args.mode == "train" or need_decoder_val:
            print(f"[sep_v1] val_images_root={val_images_root}, val_list={args.val_list}", flush=True)
            dataset_val = advCheX_hyp_multi_grade_stage_sep_v1(
                images_path=val_images_root,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
            )
        else:
            dataset_val = None

        dataset_test = advCheX_hyp_multi_grade_stage_sep_v1(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
        )
        diseases = label_names
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_multi_stage_v1":
        # ordinal 0~2 -> 2 thresholds (>=1, >=2)
        args.num_class = 2
        label_names = [">=1", ">=2"]
        if args.mode == "train":
            dataset_train = advCheX_hyp_multi_stage_v1(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                few_shot=args.few_shot,
            )
            dataset_val = advCheX_hyp_multi_stage_v1(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
            )
        else:
            dataset_train = None
            dataset_val = None

        dataset_test = advCheX_hyp_multi_stage_v1(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
        )
        diseases = label_names
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)

    elif args.data_set == "advCheX_hyp_multi_stage_v2":
        # ordinal 0~2 -> 2 thresholds (>=1, >=2)
        args.num_class = 2
        label_names = [">=1", ">=2"]
        if args.mode == "train":
            dataset_train = advCheX_hyp_multi_stage_v2(
                images_path=args.data_dir,
                file_path=args.train_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="train",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
                few_shot=args.few_shot,
            )
            dataset_val = advCheX_hyp_multi_stage_v2(
                images_path=args.data_dir,
                file_path=args.val_list,
                augment=build_transform_classification(
                    normalize=args.normalization,
                    mode="valid",
                    crop_size=args.input_size,
                    resize=args.img_size
                ),
            )
        else:
            dataset_train = None
            dataset_val = None

        dataset_test = advCheX_hyp_multi_stage_v2(
            images_path=args.data_dir,
            file_path=args.test_list,
            augment=build_transform_classification(
                normalize=args.normalization,
                mode="test",
                crop_size=args.input_size,
                resize=args.img_size
            ),
        )
        diseases = label_names
        classification_engine(args, model_path, output_path, diseases,
                             dataset_train, dataset_val, dataset_test)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
