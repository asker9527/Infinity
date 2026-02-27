"""
利用VAR生成数据，增强训练数据的多样性。
生成是批量生成，将数据的后n个Scale用于增强，一次生成B*n个数据。
"""
import os
import random
import torch
from torchvision.utils import save_image  # 引入保存图片的工具
from tools.run_infinity import *
from tools.diy_tools import get_models
from infinity.dataset.RS_datasets import get_RS_datasets, get_class2label
from torch.utils.data import DataLoader

def set_number_of_samples_per_class(nums_per_class, strategy='balance', target_num=500, fixed_add=100):
    """
    基于样本分布，设置每个类别的生成样本数量。
    :param nums_per_class: 每个类别的当前样本数量列表。
    :param strategy: 增强策略 ('balance' 长尾平衡 / 'fixed' 固定增加)
    :param target_num: balance策略下，期望每个类别达到的目标总数
    :param fixed_add: fixed策略下，每个类别固定增加的数量
    :return: 长度与类别数相同的列表，表示每个类别需要生成的数量
    """
    if strategy == 'balance':
        # 将所有类别的样本数补充到 target_num，如果已经超过则不生成 (0)
        return [max(0, target_num - num) for num in nums_per_class]
    elif strategy == 'fixed':
        # 每个类别固定增加 fixed_add 个样本
        return [fixed_add for _ in nums_per_class]
    else:
        raise ValueError("Unsupported strategy")

class generated_data_filter():
    """
    对生成的数据进行过滤，确保其质量。
    :param mode: 过滤模式 ('full', 'clip_similarity', 'uncertainty')
    """
    def __init__(self, mode='full', similarity_threshold=0.25):
        self.mode = mode
        self.similarity_threshold = similarity_threshold

    def filter(self, generated_image, prompt):
        """
        根据提示词和生成的图像进行过滤。返回 True 表示保留，False 表示丢弃。
        """
        if self.mode == 'full':
            return True  # 修正：'full' 模式应该默认通过，而不是 False

        elif self.mode == 'uncertainty':
            # TODO: 替换为实际的分类模型推理代码
            predicted_label = "dummy_label" # classify_image(generated_image)
            if predicted_label != prompt:
                return False

        elif self.mode == 'clip_similarity':
            # TODO: 替换为实际的CLIP相似度计算代码
            similarity_score = 1.0 # compute_clip_similarity(generated_image, prompt)
            if similarity_score < self.similarity_threshold:
                return False
                
        return True

    def evaluate(self, generated_image, prompt):
        """
        计算生成图像的质量分数。
        """
        if self.mode == 'uncertainty':
            # TODO: 替换为实际逻辑
            predicted_label, uncertainty_score = prompt, 0.1 # classify_image_with_uncertainty(...)
            if predicted_label != prompt:
                return 0.0
            return 1.0 - uncertainty_score
            
        elif self.mode == 'clip_similarity':
            # TODO: 替换为实际逻辑
            similarity_score = 0.9 # compute_clip_similarity(...)
            return similarity_score
            
        return 1.0 


if __name__ == "__main__":
    # 配置路径
    base_dir = "/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1"
    train_path = f"{base_dir}/Infinity/data/Asker9527/Remote_Sense_Datasets/DIOR/train"
    test_path = f"{base_dir}/Infinity/data/Asker9527/Remote_Sense_Datasets/DIOR/test"
    synthetic_save_base = f"/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/Remote_Sense_Datasets/DIOR_Synthetic"
    
    # 1. 加载数据
    train_dataset, test_dataset = get_RS_datasets(train_path, test_path)
    print("train dataset size:", len(train_dataset))
    print("test dataset size:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 2. 加载模型
    personal_data_path = '/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1'
    sft_models_path = '/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/debug_experiment022611/ar-ckpt-giter000K-ep0-iter900-last.pth'
    sft_models_path='/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1/models/FoundationVision/Infinity/infinity_125M_256x256.pth'
    vae, infinity, text_tokenizer, text_encoder, args = get_models(personal_data_path, sft_models_path, config=None)

    # 3. 确定生成数据的参数
    # 假设 get_class2label 返回如 {0: 'airplane', 1: 'airport', ...}
    class2label = get_class2label(train_path.split('/')[-2].lower())  
    label2class = {v: k for k, v in class2label.items()}  
    nums_per_class = train_dataset.get_samples_per_class()  
    
    # 选择增强策略（这里示例为每个类补充到平衡，或固定加100）
    add_nums_per_class = set_number_of_samples_per_class(nums_per_class, strategy='fixed', fixed_add=100) 

    # 提取共有参数
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-1.0))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    data_filter = generated_data_filter(mode='uncertainty') # 过滤器只需实例化一次

    # 4. 开始生成数据
    for class_idx, num_to_generate in enumerate(add_nums_per_class):
        if num_to_generate <= 0:
            continue
        class_name = label2class[class_idx]
        prompt = f'a photo of a {class_name}'  # 根据实际情况调整提示词格式
        
        # 建立类别专属的文件夹，保持与原数据集结构一致
        class_save_path = os.path.join(synthetic_save_base,str(class_idx))
        os.makedirs(class_save_path, exist_ok=True)

        valid_count = 0
        attempts = 0
        max_attempts = num_to_generate * 5  # 设置最大尝试次数，防止过滤器过于严格导致死循环

        # 使用 while 循环确保生成足够数量的合格数据
        while valid_count < num_to_generate and attempts < max_attempts:
            attempts += 1
            save_filename = os.path.join(class_save_path, f"synth_{valid_count:05d}.jpg")
            # if os.path.exists(save_filename):
            #     continue
            seed = random.randint(0, 1000000)  # 每次生成前随机设置seed，增加多样性
            # NOTE: 如果此处改为批量生成 (Batch size = B)，请将其替换为 gen_batch_img
            # 并在下方使用 for img in batch_images: 进行过滤和保存
            generated_image = gen_one_img(
                infinity, 
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                g_seed=seed,  # 每次尝试改变seed，保证生成多样性
                cfg_list=3.0,
                tau_list=1.0,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
            )
            
            # 过滤
            # if not data_filter.filter(generated_image, prompt):
            #     continue  

            # 根据 generated_image 的类型（Tensor 还是 PIL Image）选择保存方式
            # 假设 gen_one_img 返回的是归一化到 [0, 1] 的 PyTorch Tensor:
            generated_image_np = generated_image.cpu().numpy()
            if generated_image_np.shape[2] == 3:  
                generated_image_np = generated_image_np[..., ::-1]
            result_image = Image.fromarray(generated_image_np.astype(np.uint8))
            result_image.save(save_filename)
            print(f"Saved {save_filename} (Attempt {attempts}, Valid {valid_count + 1}/{num_to_generate})")
            
            valid_count += 1
            
        if attempts >= max_attempts:
            print(f"Warning: Reached max attempts for {prompt}. Generated {valid_count}/{num_to_generate}.")