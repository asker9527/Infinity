import cv2
import torch
def decoder_img(vae, summed_codes, save_path='./output/tmp.jpg'):
    img = vae.decode(summed_codes.squeeze(-3))
    # [Post-processing] 反归一化并转为 uint8 图像
    img = (img + 1) / 2
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    generated_img = img[0]
    cv2.imwrite(save_path, generated_img.cpu().numpy())

# draw attention map with topk paths
def extract_topk_paths(attn_map, k=1):
    """
    attn_map: [Lq, Lk]
    return: list of arrays, each [Lq]
    """
    import numpy as np

    topk_idx = np.argsort(attn_map, axis=-1)[:, -k:]  # [Lq, k]
    return [topk_idx[:, i] for i in range(k)]

# def draw_attention_heads(attn_maps, save_path, n_cols=4, title_prefix='Head', if_tok=False):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import math
#     from matplotlib.colors import LogNorm

#     H = attn_maps.shape[0]
#     n_rows = math.ceil(H / n_cols)

#     fig, axes = plt.subplots(
#         n_rows, n_cols,
#         figsize=(4 * n_cols, 3 * n_rows),
#         squeeze=False
#     )

#     # 避免 0 导致 log 爆炸
#     eps = 1e-6
#     vmin = max(attn_maps[attn_maps > 0].min(), eps)
#     vmax = attn_maps.max()

#     for i in range(n_rows * n_cols):
#         ax = axes[i // n_cols][i % n_cols]
#         if i < H:
#             im = ax.imshow(
#                 attn_maps[i],
#                 cmap='hot',      # 你现在这个就很好
#                 norm=LogNorm(vmin=vmin, vmax=vmax),
#                 aspect='auto'
#             )
#             ax.set_title(f'{title_prefix} {i}')
#             if if_tok:
#                 path = extract_topk_paths(attn_maps[i], k=1)
#                 for p in path:
#                     ax.plot(p, np.arange(len(p)), color='cyan', linewidth=1)
#         ax.axis('off')

#     fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
#     plt.savefig(save_path, dpi=300)
#     plt.close()



def draw_attention_heads(attn_maps, save_path, n_cols=4, title_prefix='Head', if_tok=False, fixed_vmin=0.0, fixed_vmax=1.0, use_log=False):
    """
    绘制多头注意力图

    Parameters:
    -----------
    attn_maps : np.ndarray, shape [H, Q, K]
        注意力权重矩阵，如果 Q < K (KV Cache场景)，会自动 Pad 成 [H, K, K] 方便显示。
    save_path : str
        保存路径
    n_cols : int
        每行显示多少个 head
    title_prefix : str
        子图标题前缀
    if_tok : bool
        是否绘制 token 路径
    fixed_vmin, fixed_vmax : float
        固定色阶范围
    use_log : bool
        是否使用对数色阶
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import copy
    from matplotlib.colors import LogNorm

    H, Q, K = attn_maps.shape

    # # KV Cache Pad 成方阵
    # if Q < K:
    #     padded_maps = np.full((H, K, K), 0.0)  # 使用 0 填充
    #     padded_maps[:, -Q:, :] = attn_maps
    #     display_maps = padded_maps
    #     Q = K
    # else:
    #     display_maps = attn_maps
    display_maps = attn_maps

    n_rows = math.ceil(H / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False, constrained_layout=True)

    # 设置 cmap，将 NaN 显示为浅灰色
    cmap = copy.copy(plt.cm.hot)
    cmap.set_bad(color="#040000")

    images = []

    # 选择 Norm
    norm = LogNorm(vmin=max(fixed_vmin, 1e-6), vmax=fixed_vmax) if use_log else None

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        if i < H:
            current_map = display_maps[i]
            if not use_log:
                im = ax.imshow(current_map, cmap=cmap, vmin=fixed_vmin, vmax=fixed_vmax,
                            interpolation='nearest', aspect='equal')
            else:
                im = ax.imshow(current_map, cmap=cmap, norm=norm,
                            interpolation='nearest', aspect='equal')
            images.append(im)
            ax.set_title(f'{title_prefix} {i}', fontsize=10)

            # 绘制 Token 路径
            if if_tok:
                original_map = attn_maps[i]
                path = extract_topk_paths(original_map, k=3)
                offset_y = K - original_map.shape[0]
                for p in path:
                    ax.plot(p, np.arange(len(p)) + offset_y, color='cyan', linewidth=1, alpha=0.8)

        # 隐藏刻度
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余子图
    for j in range(H, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')

    # 统一 colorbar
    if images:
        fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention maps saved to {save_path} (Reshaped to {K}x{K})")




# draw_attention_heads(attn_weight[0].cpu().numpy(), save_path='./Infinity/outputs/attn_heads.jpg')

def self_attn_with_visualize(q, k, v, scale, save_path=None, if_tok=False, use_log=False, scale_size = None):
    # 1. 计算 Q @ K.T
    # q: [B, H, L, D], k: [B, H, L, D] -> attn: [B, H, L, L]
    B, H, L, C = q.shape
    if scale_size is not None:
        L_cat = int(0.5*scale_size*scale_size)
        attn_weight = torch.empty(
            (B, H, L, L),
            device=q.device,
            dtype=q.dtype
        )

        attn_left = torch.matmul(
            q,
            k[:, :, :-L_cat, :].transpose(-2, -1)
        ) * scale
        attn_weight[:, :, :, :-L_cat] = attn_left
        attn_weight[:, :, :, -L_cat:] = attn_left[:, :, :, -L_cat:].flip(-2, -1)

        del attn_left  # 立刻释放
    else:
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # # 2. 处理 Mask (如果有)
    # if attn_bias_or_two_vector is not None:
    #     # 假设 attn_bias_or_two_vector 是加性 mask (float)
    #     # 如果是 boolean mask，你需要用 masked_fill
    #     attn_weight = attn_weight + attn_bias_or_two_vector
    
    # 3. Softmax
    attn_weight = attn_weight.softmax(dim=-1)
    if save_path is not None:
        draw_attention_heads(attn_weight[0].cpu().numpy(), save_path=save_path, if_tok=if_tok, use_log=use_log)

    # 4. 计算 Output: Attn @ V
    # [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]
    oup = torch.matmul(attn_weight, v)
    
    # 5. 恢复形状
    oup = oup.transpose(1, 2).reshape(B, L, -1)
    return oup


# TODO: 完成prompt2prompt功能
def prompt2prompt(source_prompt, target_prompt, model, image):
    """
    实现prompt to prompt的功能
    参考论文：Prompt-to-Prompt Image Editing with Cross Attention Control
    论文链接：https://arxiv.org/abs/2208.01626
    主要思路是在生成过程中，控制不同prompt之间的注意力映射，从而实现对图像内容的精细编辑。
    具体实现步骤包括：
    1. 提取原始prompt和目标prompt的文本嵌入表示。
    2. 在生成过程中，计算注意力权重，并根据原始prompt和目标prompt的差异调整这些权重。
    3. 通过调整注意力权重，实现对图像内容的局部修改，同时保持整体风格和结构的一致性。
    4. 最终生成符合目标prompt要求的图像。
    该方法允许用户通过修改文本描述，灵活地编辑生成图像的内容，而无需重新训练模型。
    """
    # 提取文本嵌入
    source_embedding = model.get_text_embedding(source_prompt)
    target_embedding = model.get_text_embedding(target_prompt)

    # 生成图像时调整注意力权重
    generated_image = model.generate_with_attention_control(
        image,
        source_embedding,
        target_embedding
    )

    return generated_image

def generate_with_attention_control(model, image, source_embedding, target_embedding):
    """
    在生成过程中，控制注意力权重以实现prompt to prompt的功能。
    """
    # 伪代码示例，具体实现取决于模型架构
    for layer in model.layers:
        attn_weights = layer.compute_attention_weights(image, source_embedding)
        adjusted_weights = adjust_attention_weights(attn_weights, source_embedding, target_embedding)
        image = layer.apply_attention(image, adjusted_weights)
    return image

def adjust_attention_weights(attn_weights, source_embedding, target_embedding):
    """
    根据源嵌入和目标嵌入调整注意力权重。
    """
    # 伪代码示例，具体调整策略可以根据需求设计
    adjustment = target_embedding - source_embedding
    adjusted_weights = attn_weights + adjustment.unsqueeze(1).unsqueeze(2)
    return adjusted_weights




def cross_attn_with_visualize(q, k, v, scale, attn_bias_or_two_vector=None):
    import math
    # 1. 计算 Q @ K.T
    # q: [B, H, Lq, D], k: [B, H, Lk, D] -> attn: [B, H, Lq, Lk]
    B, H, Lq, C = q.shape
    Lk = k.shape[2]
    attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # # 2. 处理 Mask (如果有)
    # if attn_bias_or_two_vector is not None:
    #     # 假设 attn_bias_or_two_vector 是加性 mask (float)
    #     # 如果是 boolean mask，你需要用 masked_fill
    #     attn_weight = attn_weight + attn_bias_or_two_vector
    
    # 3. Softmax
    attn_weight = attn_weight.softmax(dim=-1)
    draw_attention_map(attn_weight[0,0].cpu().numpy(), save_path='./output/cross_attn_map.jpg')

    # 4. 计算 Output: Attn @ V
    # [B, H, Lq, Lk] @ [B, H, Lk, D] -> [B, H, Lq, D]
    oup = torch.matmul(attn_weight, v)
    
    # 5. 恢复形状
    oup = oup.transpose(1, 2).reshape(B, Lq, C)
    return oup