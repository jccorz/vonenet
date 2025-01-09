import vonenet
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np



from vonenet import get_model
from vonenet.activation import ActivationExtractor


def main():



    # Load model

    # Create random input
    image_path = '/Users/jcc/Desktop/v1/vonenet/rdm_grating'  # 替换为实际图片路径
    dataset = torchvision.datasets.ImageFolder(image_path,
                                               torchvision.transforms.Compose([
                                                   torchvision.transforms.Resize(224),
                                                   torchvision.transforms.CenterCrop(224),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                    std=[0.5, 0.5, 0.5]),
                                               ]))
    bsize = 16
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_iterator = iter(data_loader)
    x, _ = next(dataloader_iterator)






    # x = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    # Choose blocks for hook
    layer_names = [
        'vone_block',  # V1 layer
        'model.V2.conv1',
        'model.V2.conv3',
        'model.V2',    # V2 layer
        'model.V4',    # V4 layer
        'model.IT'     # IT layer
    ]

    n_sample_col = {
        'vone_block': 700,
        'model.V2.conv1': 600,
        'model.V2.conv3': 600,
        'model.V2': 500,
        'model.V4': 400,    # V4 layer
        'model.IT': 300,     # IT layer
    }


    # Extract activations from non-noisy data
    model_no_noise = vonenet.get_model(model_arch='cornets', pretrained=True, noise_mode="neuronal").module
    model_no_noise.vone_block.noise_mode = None
    extractor_nonoise = ActivationExtractor(model_no_noise, layer_names)
    activations_nonoise = extractor_nonoise.extract(x)

    # Set sampling columns and store their indices
    ind_sample_col = {}
    set_percentile = 0.5
    for layer_name, activation in activations_nonoise.items():
        # 获取该层的激活值并转换为 NumPy 数组
        # 将激活值展平成二维数组，形状为 (batch_size, num_features)
        activation_reshaped = activation.detach().numpy().reshape(bsize, -1)
        # 获取要采样的列数
        num_sample_cols = n_sample_col.get(layer_name, 0)

        if num_sample_cols > 0 and num_sample_cols <= activation_reshaped.shape[1]*set_percentile:
            percentile_threshold = np.percentile(np.mean(activation_reshaped, axis=0), set_percentile)
            # 随机选择列的索引
            selected_indices = np.where(np.mean(activation_reshaped, axis=0) >= percentile_threshold)[0]
            sampled_indices = np.random.choice(selected_indices, num_sample_cols, replace=False)

            # 存储采样的列
            ind_sample_col[layer_name] = sampled_indices
        else:
            print(f"层 {layer_name} 的采样列数无效或超过总特征数。")

    count = 0
    sample_size = 1010
    dataloader_iterator = iter(data_loader)  # 放在loop里面意味着我们无法操控读取什么样的图片
    x, _ = next(dataloader_iterator)
    x_mirr, _ = next(dataloader_iterator)
    # Extract activations from multiple layers
    model = vonenet.get_model(model_arch='cornets', pretrained=True, noise_mode="neuronal").module

    while True:
        if count > sample_size:
            break
        else:
            count += bsize
            extractor = ActivationExtractor(model, layer_names)
            activations = extractor.extract(x)
            activations_nonoise = extractor_nonoise.extract(x_mirr)

            # Each block's activation
            for layer_name, activation in activations.items():
                # resample = activation.reshape(activation.shape[0], -1)[:, ind_sample_col[layer_name]] - \
                #            activations_nonoise[layer_name].reshape(activations_nonoise[layer_name].shape[0], -1)[:, ind_sample_col[layer_name]]
                resample = activation.reshape(activation.shape[0], -1)[:, ind_sample_col[layer_name]]
                vonenet.save_to_csv(resample.detach().numpy(),layer_name)

if __name__ == '__main__':
    main()