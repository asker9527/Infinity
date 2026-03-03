import argparse
import torch
import torch.nn.functional as F
import os
import PIL.Image as Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer


def draw_gernerated_image(self, logits_BLV,src ,vae_scale_schedule, g_it, path=None, mode='train'):
    logits_BlV = logits_BLV.clone() # [4, 521, 64]
    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
    idx_Bld_full = torch.argmax(logits_BlV, dim=-1)  # [4, seq_len]
    idx_Bld_full = idx_Bld_full.reshape(tmp_bs, tmp_seq_len, -1)
    B=logits_BLV.shape[0]
    start = 0 
    summed_codes = 0
    for i,si in enumerate(vae_scale_schedule):
        scale = si[-1]
        end = start + si[-1]*si[-2]
        idx_Bld = idx_Bld_full[:,start:end,:].reshape(B, scale,scale, -1)
        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = self.vae_local.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=self.vae_local.quantizer.z_interplote_up)
        start = end

    img = self.vae_local.decode(summed_codes.squeeze(-3))
    img = (img + 1) / 2
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8) # [B, H, W, 3]
    img_src = (src + 1) / 2
    img_src = img_src.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)

    # for i in range(img.shape[0]):
    #     Image.fromarray(img[i].cpu().numpy()).save(os.path.join(path, f'debug_infinity_output_{g_it}_{i}.png'))
    #     Image.fromarray(img_src[i].cpu().numpy()).save(os.path.join(path, f'debug_infinity_input_{g_it}_{i}.png'))

    # --- 替换开始 ---
    import torchvision.utils as vutils
    
    # 1. 将数据转回 [B, 3, H, W] 以便使用 make_grid
    # (因为前面代码已经变成了 [B, H, W, 3]，这里临时转回来方便拼接)
    vis_src = img_src.permute(0, 3, 1, 2)
    vis_tgt = img.permute(0, 3, 1, 2)
    
    # 2. 交替存入列表: [Src1, Tgt1, Src2, Tgt2, ...]
    comparison_list = []
    for s, t in zip(vis_src, vis_tgt):
        comparison_list.append(s)
        comparison_list.append(t)
    
    # 3. 拼成网格图
    # nrow=8 表示一行显示 8 张图 (即 4 对 [Src, Tgt])
    # padding=2 会在图片之间留出 2 像素的黑色边框
    grid = vutils.make_grid(torch.stack(comparison_list), nrow=2, padding=2, normalize=False)
    
    # 4. 转回 [H, W, 3] 并保存
    ndarr = grid.permute(1, 2, 0).contiguous().cpu().numpy()
    Image.fromarray(ndarr).save(os.path.join(path, f'{mode}_comparison_{g_it}.png'))
    # --- 替换结束 ---

import random
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
personal_code_path = '/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1'
personal_data_path = '/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1'
import sys
import os

def get_models(personal_data_path, sft_models_path=None, config=None):
    models_dir = personal_data_path + '/models'
    if sft_models_path:
        model_path = sft_models_path
    else:
        model_path= os.path.join(models_dir, 'FoundationVision/Infinity/infinity_125M_256x256.pth')
    vae_path=os.path.join(models_dir,'FoundationVision/Infinity/infinity_vae_d16.pth')
    text_encoder_ckpt = os.path.join(models_dir, 'google/flan-t5-xl')
    print(text_encoder_ckpt)
    args=argparse.Namespace(
        pn='0.06M',         # 125M: 0.06M; 2B: 1M
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=16,        # 125M: 16; 2B: 32
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_layer12',  # infinity_layer12, infinity_2b, 2bc8
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0, 
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir= models_dir,
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        enable_model_cache = False, # add parameter
        # save_file='tmp.jpg'
    )
    # updata args with config
    if config:
        for key, value in config.items():
            setattr(args, key, value)

    text_tokenizer, text_encoder = load_tokenizer(t5_path = args.text_encoder_ckpt,)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    return vae, infinity, text_tokenizer, text_encoder, args

if __name__ == "__main__":
    # Debug get_models function
    personal_code_path = '/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1'
    personal_data_path = '/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1'
    vae, infinity, text_tokenizer, text_encoder = get_models(personal_data_path)
    print(vae)
