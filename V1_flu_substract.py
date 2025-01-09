import vonenet
import torchvision
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_activation(name, actdict):
    def hook(model, input, output):
        actdict[name] = output.detach()
    return hook


def get_new_data(X, v1_model, v1_layer=None, v2_layer=None, V2_position=None, V2_size_side=None):
    actdict = {}
    hooks = []
    hooks.append(v1_model.vone_block.output.register_forward_hook(get_activation('V1', actdict)))
    # hooks.append(v1_model.model.V2.output.register_forward_hook(get_activation('V2', actdict)))
    hooks.append(v1_model.model.V2.output.register_forward_hook(get_activation('V2', actdict)))
    hooks.append(v1_model.model.V2.conv1.register_forward_hook(get_activation('V2inter', actdict)))

    v1_model.eval()
    with torch.no_grad():
        v1_model(X)  # Perform the forward pass

    # Remove hooks to prevent memory leaks
    for hook in hooks:
        hook.remove()

    V1_np = actdict['V1'].cpu().numpy().astype('float16')
    V2_np = actdict['V2'].cpu().numpy().astype('float16')
    V2int_np = actdict['V2inter'].cpu().numpy().astype('float16')


    # 计算 V1 和 V2 的位置和采样大小
    V1_position = (V2_position[0] * 2-1, V2_position[1] * 2-1)
    V1_size_side = (V2_size_side) * 3


    # 从 V1_position 和 V2_position 开始对 m_ref_V1 和 m_ref_V2 进行采样
    if v1_layer is not None and v2_layer is not None:
        V1_sample = V1_np[:, v1_layer, V1_position[0]:V1_position[0] + V1_size_side,
                    V1_position[1]:V1_position[1] + V1_size_side]
        V2int_sample = V2int_np[:, v1_layer, V1_position[0]:V1_position[0] + V1_size_side,
                    V1_position[1]:V1_position[1] + V1_size_side]
        V2_sample = V2_np[:, v2_layer, V2_position[0]:V2_position[0] + V2_size_side,
                    V2_position[1]:V2_position[1] + V2_size_side]

    else:
        V1_sample = V1_np[:, :, V1_position[0]-1:V1_position[0] + V1_size_side-1,
                    V1_position[1]-1:V1_position[1] + V1_size_side-1]
        V2int_sample = V2int_np[:, :, V1_position[0]-1:V1_position[0] + V1_size_side-1,
                    V1_position[1]-1:V1_position[1] + V1_size_side-1]
        V2_sample = V2_np[:, :, V2_position[0]:V2_position[0] + V2_size_side,
                    V2_position[1]:V2_position[1] + V2_size_side]
    V1_reshaped = V1_sample.reshape(V1_sample.shape[0], -1)
    V2int_reshaped = V2int_sample.reshape(V2int_sample.shape[0], -1)
    V2_reshaped = V2_sample.reshape(V2_sample.shape[0], -1)

    # print(V1_reshaped.shape, V2_reshaped.shape)
    # 将 V1 和 V2 样本重塑为一维数组
    # v1s = V1_reshaped[:, :n_src].astype('float32')
    # v1t = V1_reshaped[:, n_src:n_src + n_trg].astype('float32')
    # v2 = V2_reshaped[:, :n_trg].astype('float32')
    return V1_reshaped, V2_reshaped,V2int_reshaped

# def append_to_csv(df, csv_path):
#     """ Append a DataFrame to a CSV file. """
#     if os.path.exists(csv_path):
#         df.to_csv(csv_path, mode='a', header=False, index=False)
#     else:
#         df.to_csv(csv_path, mode='w', header=False, index=False)
# def load_existing_csv(csv_path):
#     """ Load existing data from a CSV file. """
#     if os.path.exists(csv_path):
#         return pd.read_csv(csv_path, header=None)  # Load without headers
#
#     return pd.DataFrame()

def save_data_to_csv(v1s, v1t, v2,v2s, output_dir):
    """ Save the data to CSV files in the specified output directory, appending if necessary. """
    # Define filenames and corresponding data
    file_data_pairs = [
        ('v1.csv', v1s),
        ('v1t.csv', v1t),
        ('v2int.csv', v2),
        ('v2.csv', v2s)
    ]

    for file_name, data in file_data_pairs:
        file_path = os.path.join(output_dir, file_name)
        # Convert new data to DataFrame
        df = pd.DataFrame(data)
        # Append new data to CSV (mode 'a' for append, header=False to avoid adding the header repeatedly)
        df.to_csv(file_path, mode='a', header=False, index=False)


def csv_data_size(csv_path):
    """ Return the number of rows in the CSV file. """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.shape[0]
    return 0


import numpy as np


def filter_and_random_select(data, n_src, n_trg, top_percent=0.25):
    """
    先筛选出均值较大的列，再随机选出一定数量的列。
    示例: 从 V2_sp 中筛选
    n_src = 5  # 源列数量
    n_trg = 3  # 目标列数量
    col_sp = filter_and_random_select(V2_sp, n_src, n_trg)

    参数:
    data: 2D NumPy 数组 (V2_sp)
    n_src: 随机选择的源列数量
    n_trg: 随机选择的目标列数量
    top_percent: 用于筛选均值较大的百分比 (默认25%)
    """

    # 1. 计算每一列的均值
    col_means = np.mean(data, axis=0)
    # 2. 根据均值筛选出前 top_percent 百分位的列
    threshold = np.percentile(col_means, (1 - top_percent) * 100)
    filtered_columns = np.where(col_means >= threshold)[0]
    print(filtered_columns.shape)

    # 3. 随机选择 n_src + n_trg 个列
    col_sp = np.random.choice(filtered_columns, n_src + n_trg, replace=False)
    return col_sp


def main(n_src=200,n_trg=150,n_data=200):
    data_path = '/Users/jcc/Desktop/v1/vonenet/val_real'
    bsize = 16
    crop = 256  # 48  256
    px = 224  # 32  224
    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    dataset = torchvision.datasets.ImageFolder(data_path,
                                               torchvision.transforms.Compose([
                                                   torchvision.transforms.Resize(crop),
                                                   torchvision.transforms.CenterCrop(px),
                                                   torchvision.transforms.ToTensor(),
                                                   normalize,
                                               ]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    v1_model = vonenet.get_model(model_arch='cornets', pretrained=True, noise_mode="neuronal").module
    # v1_model_mean = vonenet.get_model(model_arch='cornets', pretrained=True, noise_mode="neuronal").module
    v1_model.vone_block.noise_mode = "None"
    print(v1_model.vone_block.noise_mode,v1_model.vone_block.noise_level,v1_model.vone_block.noise_scale)


    # 20, 254, 255, 271(Vague)
    # V1_layer = 20
    # 22, 107, 113, 108(Vague)
    # V2_layer = 22
    V2_position = (5, 5)
    V2_size_side = 1
    col_sp_v1=[]
    col_sp_v2=[]
    v1_mean = None
    v2_mean = None

    data_not_enough = True
    col_is_selected = False

    output_dir = '/Users/jcc/Desktop/v1/vonenet'
    v1s_path = os.path.join(output_dir, 'v1s.csv')
    v1t_path = os.path.join(output_dir, 'v1t.csv')
    v2_path = os.path.join(output_dir, 'v2.csv')
    # Calculate no noise mean
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_iterator = iter(data_loader)
    X, _ = next(dataloader_iterator)
    v1_m, v2_m, v2int_m = get_new_data(X, v1_model, V2_position=V2_position, V2_size_side=V2_size_side)


    v1_model = vonenet.get_model(model_arch='cornets', pretrained=True, noise_mode="neuronal").module

    while True:
        dataloader_iterator = iter(data_loader)
        X, _ = next(dataloader_iterator)

        v1, v2, v2int = get_new_data(X, v1_model, V2_position=V2_position, V2_size_side=V2_size_side)
        # Sampling cross layers
        if not col_is_selected:
            col_sp_v1 = filter_and_random_select(v1, 4000, 0, top_percent=1)
            col_sp_v2 = filter_and_random_select(v2, 64, 64, top_percent=1)
            col_is_selected = True
        # col_sp_v1 = np.random.choice(v1.shape[1], n_src+n_trg, replace=False)
        # col_sp_v2 = np.random.choice(v2.shape[1], n_src+n_trg, replace=False)
        # v1 = v1-v1_mean
        # v2 = v2-v2_mean
        # v1 = v1[:, col_sp_v1]
        # v2 = v2[:, col_sp_v2]

        v1s = v1 - v1_m
        v1t = v1[:,1:1]
        v2s = v2 - v2_m
        v2t = v2int -v2int_m
        # Check the size of the CSV files
        if (csv_data_size(v1s_path) >= n_data and
                csv_data_size(v2_path) >= n_data):
            print(
                f"Data sizes are sufficient: v1s: {csv_data_size(v1s_path)}, v1t: {csv_data_size(v1t_path)}, v2: {csv_data_size(v2_path)}")
            break
        else:
            save_data_to_csv(v1s, v1t, v2t ,v2s, output_dir)
            print(f"Data insufficient, collecting more data... Col select: {[]}")


if __name__ == '__main__':
    main(4400, 150, 2000)

