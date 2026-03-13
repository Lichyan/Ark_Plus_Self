import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.swin_transformer import SwinTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed

from torch.hub import load_state_dict_from_url
from timm.models.helpers import load_state_dict

from functools import partial
import simmim
#from upernet_swin_transformer import UperNet_swin
from convnext import ConvNeXt
from resnet import ResNet50
from utils import load_swin_pretrained
from lora import inject_lora, freeze_non_lora_parameters

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - older torch versions
    add_safe_globals = None

if add_safe_globals is not None:
    try:
        add_safe_globals([np.core.multiarray.scalar])
    except AttributeError:
        pass

def build_classification_model(args):
    model = None
    print("Creating model...")
    if args.pretrained_weights is None or args.pretrained_weights =='':
        print('Loading pretrained {} weights for {} from timm.'.format(args.init, args.model_name))
        if args.model_name.lower() == "vit_base":
            if args.init.lower() =="random":
                if args.input_size == 448:
                    model = VisionTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                else:
                    model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                # model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_base_patch16_224_in21k', num_classes=args.num_class, pretrained=True)  
            elif args.init.lower() =="sam":
                model = timm.create_model('vit_base_patch16_224_sam', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                #model = timm.create_model('vit_base_patch16_224_dino', num_classes=args.num_class, pretrained=True) #not available in current timm version
                url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="beit":
                model = timm.create_model('beit_base_patch16_224', num_classes=args.num_class, pretrained=True)

        elif args.model_name.lower() == "vit_small":
            if args.init.lower() =="random":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_small_patch16_224_in21k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                #model = timm.create_model('vit_small_patch16_224_dino', num_classes=args.num_class, pretrained=True)
                model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_small_patch16_224', num_classes=args.num_class, pretrained=True)           
        
        elif args.model_name.lower() == "swin_large":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            
        elif args.model_name.lower() == "swin_large_384":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size, 
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))

        elif args.model_name.lower() == "swin_base": 
            if args.init.lower() =="random":
                if args.input_size == 448:
                    model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                else:
                    model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_21kto1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=True)
            
        elif args.model_name.lower() == "swin_tiny": 
            if args.init.lower() =="random":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=True)
        
        elif args.model_name.lower() == "convx_base":
            if args.init.lower() =="random":
                model = timm.create_model('convnext_base_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('convnext_base.fb_in1k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('convnext_base_in22k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21kto1k":
                model = timm.create_model('convnext_base_in22ft1k', num_classes=args.num_class, pretrained=True)
        elif args.model_name.lower() == "resnet50":
            if args.init.lower() =="random":
                model = ResNet50(num_classes=args.num_class)
        
    else:
        print("Creating model from pretrained weights: "+ args.pretrained_weights)
        if args.model_name.lower() == "vit_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            else:
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                load_pretrained_weights(
                    model,
                    args.init.lower(),
                    args.pretrained_weights,
                    keep_head=getattr(args, "keep_head", False),
                )
            
        elif args.model_name.lower() == "vit_small":
            model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.default_cfg = _cfg()
            load_pretrained_weights(
                model,
                args.init.lower(),
                args.pretrained_weights,
                keep_head=getattr(args, "keep_head", False),
            )
            
        elif args.model_name.lower() == "swin_large":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            load_pretrained_weights(
                model,
                args.init.lower(),
                args.pretrained_weights,
                args.key,
                args.scale_up,
                getattr(args, "keep_head", False),
            )
            
        elif args.model_name.lower() == "swin_large_384":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size, 
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            load_pretrained_weights(
                model,
                args.init.lower(),
                args.pretrained_weights,
                args.key,
                args.scale_up,
                getattr(args, "keep_head", False),
            )
        
        elif args.model_name.lower() == "swin_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class)
                load_pretrained_weights(
                    model,
                    args.init.lower(),
                    args.pretrained_weights,
                    keep_head=getattr(args, "keep_head", False),
                )
            else:
                model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                    patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                load_pretrained_weights(
                    model,
                    args.init.lower(),
                    args.pretrained_weights,
                    args.key,
                    args.scale_up,
                    getattr(args, "keep_head", False),
                )
                
        elif args.model_name.lower() == "swin_tiny": 
            model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)
            load_pretrained_weights(
                model,
                args.init.lower(),
                args.pretrained_weights,
                keep_head=getattr(args, "keep_head", False),
            )
            
        elif args.model_name.lower() == "convx_base":
          if args.init.lower().startswith("ark"):
                model = ConvNeXt(num_classes=args.num_class,
                     depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
                load_pretrained_weights(
                    model,
                    args.init.lower(),
                    args.pretrained_weights,
                    args.key,
                    False,
                    getattr(args, "keep_head", False),
                )
          
    if model is None:
        print("Not provide {} pretrained weights for {}.".format(args.init, args.model_name))
        raise Exception("Please provide correct parameters to load the model!")

    if getattr(args, "data_set", "") in {"advCheX_hyp_multi_grade_stage_v1", "advCheX_hyp_multi_grade_stage_sep_v1"}:
        model = MultiHeadOrdinalModel(
            backbone=model,
            num_class_grade=getattr(args, "num_class_grade", 3),
            num_class_stage=getattr(args, "num_class_stage", 2),
            ordinal_mode=getattr(args, "ordinal_mode", "default"),
            sep_head_mode=getattr(args, "sep_head_mode", "flat"),
        )

    if getattr(args, "use_lora", False):
        target_tokens = [token.strip() for token in args.lora_targets.split(',') if token.strip()]
        replaced = list(inject_lora(
            model,
            target_tokens,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        ))
        if not replaced:
            print(f"[LoRA] 未匹配到需要注入的模块，请检查 --lora_targets={args.lora_targets}")
        else:
            preview = ', '.join(replaced[:5])
            if len(replaced) > 5:
                preview += f", ... (共 {len(replaced)} 层)"
            print(f"[LoRA] 已注入的模块: {preview}")
        freeze_non_lora_parameters(model, keep_head=getattr(args, "lora_train_head", True))
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LoRA] 可训练参数量: {trainable:,} / {total:,}")

    return model
    

def _extract_backbone_features(backbone, x):
    if hasattr(backbone, "forward_features"):
        feats = backbone.forward_features(x)
        if hasattr(backbone, "forward_head"):
            try:
                feats = backbone.forward_head(feats, pre_logits=True)
            except TypeError:
                feats = backbone.forward_head(feats)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        return feats
    if hasattr(backbone, "features"):
        feats = backbone.features(x)
        return torch.flatten(feats, 1)
    return backbone(x)


class MultiHeadOrdinalModel(nn.Module):
    def __init__(self, backbone, num_class_grade=3, num_class_stage=2, ordinal_mode="coral", sep_head_mode="flat"):
        super().__init__()
        self.backbone = backbone
        self.ordinal_mode = str(ordinal_mode).lower()
        self.sep_head_mode = str(sep_head_mode).lower()
        feature_dim = getattr(backbone, "num_features", None)
        if feature_dim is None:
            feature_dim = getattr(backbone, "fc", None).in_features if hasattr(backbone, "fc") else None
        if feature_dim is None:
            raise ValueError("无法推断 backbone 特征维度")

        if self.sep_head_mode == "coarse_fine":
            self.head_anyhtn = nn.Linear(feature_dim, 1)
            self.head_grade_pos = nn.Linear(feature_dim, 2)
            self.head_stage_pos = nn.Linear(feature_dim, 1)
        else:
            if self.ordinal_mode == "coral":
                self.head_grade = CoralHead(feature_dim, num_class_grade)
                self.head_stage = CoralHead(feature_dim, num_class_stage)
            else:
                self.head_grade = nn.Linear(feature_dim, num_class_grade)
                self.head_stage = nn.Linear(feature_dim, num_class_stage)

    def forward(self, x):
        feats = _extract_backbone_features(self.backbone, x)
        if self.sep_head_mode == "coarse_fine":
            logits_anyhtn = self.head_anyhtn(feats)
            logits_grade_pos = self.head_grade_pos(feats)
            logits_stage_pos = self.head_stage_pos(feats)
            return {
                "anyhtn": logits_anyhtn,
                "grade_pos": logits_grade_pos,
                "stage_pos": logits_stage_pos,
            }
        logits_grade = self.head_grade(feats)
        logits_stage = self.head_stage(feats)
        return logits_grade, logits_stage


class CoralHead(nn.Module):
    def __init__(self, in_features, num_thresholds):
        super().__init__()
        self.score = nn.Linear(in_features, 1)
        self.t1 = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(max(num_thresholds - 1, 0)))
        self.num_thresholds = num_thresholds

    def ordered_bias(self):
        if self.num_thresholds <= 1:
            return self.t1
        inc = torch.nn.functional.softplus(self.delta)
        b = [self.t1]
        run = self.t1
        for i in range(len(inc)):
            run = run + inc[i:i+1]
            b.append(run)
        return torch.cat(b, dim=0)

    def forward(self, x):
        s = self.score(x)
        b = self.ordered_bias()
        logits = s - b[None, :]
        return logits

def load_pretrained_weights(model, init, pretrained_weights, checkpoint_key=None, scale_up=False, keep_head=False):
    if pretrained_weights.startswith('https'):
        checkpoint = load_state_dict_from_url(url=pretrained_weights, map_location='cpu')
    else:
        try:
            checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError) as err:
            print(
                "[WARN] weights_only 加载失败，将回退到安全可信环境下的完整反序列化。"
                f" 原因: {err}"
            )
            checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    print(checkpoint.keys())
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']

    if init =="dino":
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in checkpoint:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = checkpoint[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    elif init =="moco_v3":
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif init == "moby":
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    # elif init == "mae":
    #     state_dict = checkpoint['model']
    elif init.startswith("ark"): 
        print("Loading {} from checkpoint...".format(checkpoint_key))
        state_dict = checkpoint[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() }
  
    else:
        print("Trying to load the checkpoint for {} at {}, but we cannot guarantee the success.".format(init, pretrained_weights))

    if scale_up:
        k_del = []
        for k in state_dict.keys():
            if "attn_mask" in k:
                k_del.append(k)
        print(f"Removing key {k_del} from pretrained checkpoint")
        for k in k_del:
            del state_dict[k]

    if not keep_head:
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del state_dict[k]
    else:
        print("Preserving classification head weights from checkpoint")
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded with msg: {}'.format(msg)) 

    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')
