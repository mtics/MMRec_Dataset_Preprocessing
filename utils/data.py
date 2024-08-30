from collections import Counter
import numpy as np
import pandas as pd
import os

from PIL import Image
import torch
import open_clip

from utils.utils import getDF, compute_modalities_similarity
from tqdm import tqdm  # 导入 tqdm 库


def get_illegal_ids_by_inter_num(data, field, max_num=None, min_num=None):
    """
    Get illegal ids by interaction number
    :param data: data frame
    :param field:
    :param max_num:
    :param min_num:
    :return:
    """

    if field is None:
        return set()
    if max_num is None and min_num is None:
        return set()

    max_num = max_num or np.inf
    min_num = min_num or -1

    ids = data[field].values
    inter_num = Counter(ids)
    ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
    print(f'Illegal {field} num: {len(ids)}')

    return ids


def filter_by_k_core(data, learner_id, course_id, min_user_num, min_item_num):
    """
    Filter data by k-core
    :param data: data frame
    :param learner_id:
    :param course_id:
    :param min_user_num:
    :param min_item_num:
    :return:
    """

    while True:
        ban_users = get_illegal_ids_by_inter_num(data, field=learner_id, max_num=None, min_num=min_user_num)
        ban_items = get_illegal_ids_by_inter_num(data, field=course_id, max_num=None, min_num=min_item_num)
        if len(ban_users) == 0 and len(ban_items) == 0:
            return

        dropped_inter = pd.Series(False, index=data.index)
        if learner_id:
            dropped_inter |= data[learner_id].isin(ban_users)
        if course_id:
            dropped_inter |= data[course_id].isin(ban_items)
        print(f'{len(dropped_inter)} dropped interactions')
        data.drop(data.index[dropped_inter], inplace=True)


def rating2inter(uid_field, iid_field, rating_field, timestamp_field, dataset_dir, output_dir, load_file,
                 min_u_num=5, min_i_num=5, split_ratios=[0.8, 0.1, 0.1]):
    """
    Convert rating data to interaction data and split it into training, validation, and test sets
    """

    # ============== 1. Filter data by k-core ==============

    # 5-core filtering
    dataset = dataset_dir.split('/')[-1]
    print(f'Filtering data by k-core for {dataset}...')
    if dataset == 'Baby':
        inter_data = pd.read_csv(os.path.join(dataset_dir, load_file), sep=',', header=None,
                                 names=[uid_field, iid_field, rating_field, timestamp_field])
    elif dataset in ['Food', 'Movie', 'Dance', 'KU']:
        inter_data = pd.read_csv(os.path.join(dataset_dir, load_file), sep=',', header=None,
                                 names=[uid_field, iid_field, timestamp_field])
        inter_data[rating_field] = 1.0

    print(f'Original data size: {inter_data.shape}')

    inter_data.dropna(subset=[uid_field, iid_field, timestamp_field], inplace=True)
    inter_data.drop_duplicates(subset=[uid_field, iid_field, timestamp_field], inplace=True)
    print(f'After removing missing values and duplicates: {inter_data.shape}')

    filter_by_k_core(inter_data, uid_field, iid_field, min_u_num, min_i_num)
    print(f'k-core filtering: {inter_data.shape}')

    # Reindex user and item ids
    inter_data.reset_index(drop=True, inplace=True)

    uni_users = inter_data[uid_field].unique()
    uni_items = inter_data[iid_field].unique()

    # user id mapping start from 0
    u_id_mapping = {uid: idx for idx, uid in enumerate(uni_users)}
    i_id_mapping = {iid: idx for idx, iid in enumerate(uni_items)}

    inter_data[uid_field] = inter_data[uid_field].map(u_id_mapping).astype(int)
    inter_data[iid_field] = inter_data[iid_field].map(i_id_mapping).astype(int)

    # Save user and item id mapping
    u_id_mapping = pd.DataFrame(list(u_id_mapping.items()), columns=['user_id', uid_field])
    i_id_mapping = pd.DataFrame(list(i_id_mapping.items()), columns=['asin', iid_field])

    u_mapping_file = os.path.join(output_dir, 'u_id_mapping.csv')
    i_mapping_file = os.path.join(output_dir, 'i_id_mapping.csv')

    u_id_mapping.to_csv(u_mapping_file, sep=',', index=False)
    i_id_mapping.to_csv(i_mapping_file, sep=',', index=False)

    print('The mapped IDs are saved!')

    # ============== 2. Split data ==============

    # Use the timestamp to split the data into training, validation, and test sets with the given ratios
    print('Splitting data...')
    tot_ratio = sum(split_ratios)
    # remove 0.0 in split_ratio
    split_ratios = [r for r in split_ratios if r > 0.0]
    split_ratios = [r / tot_ratio for r in split_ratios]
    split_ratios = np.cumsum(split_ratios)[:-1]

    split_timestamps = list(np.quantile(inter_data[timestamp_field], split_ratios))
    # Get df training dataset with unique users and items
    df_train = inter_data[inter_data[timestamp_field] <= split_timestamps[0]].copy()
    df_valid = inter_data[(inter_data[timestamp_field] >= split_timestamps[0]) & (
            inter_data[timestamp_field] <= split_timestamps[1])].copy()
    df_test = inter_data[inter_data[timestamp_field] >= split_timestamps[1]].copy()

    print(f'Train size: {df_train.shape}, Valid size: {df_valid.shape}, Test size: {df_test.shape}')

    # Save the split data
    split_label = 'split_label'
    split_file = os.path.join(output_dir, 'inter.csv')

    df_train[split_label] = 0
    df_valid[split_label] = 1
    df_test[split_label] = 2

    tmp_df = pd.concat([df_train, df_valid, df_test], axis=0)
    tmp_df = tmp_df[[uid_field, iid_field, rating_field, timestamp_field, split_label]]

    tmp_df.to_csv(split_file, sep=',', index=False)
    print('The split data is saved!')

    # Reload the split data for test
    idx_df = pd.read_csv(split_file, sep=',')
    print(f'Reload the split data: {idx_df.shape}')
    uni_users = idx_df[uid_field].unique()
    uni_items = idx_df[iid_field].unique()
    print(f'Unique users: {len(uni_users)}, Unique items: {len(uni_items)}')

    print(f'min/max user id: {min(uni_users)}/{max(uni_users)}')
    print(f'min/max item id: {min(uni_items)}/{max(uni_items)}')

    return tmp_df


def loo_split(inter_data, uid_field, iid_field, timestamp_field):
    """
    Leave-one-out splitting
    """

    print('Splitting with the Leave-one-out strategy...')

    # Construct user-item interaction dictionary for each user
    inter_data.sort_values(by=[uid_field, timestamp_field], inplace=True)
    uid_freq = inter_data.groupby(uid_field)[iid_field]
    user_inter_dict = {uid: list(inter) for uid, inter in uid_freq}

    # Construct the Splitting Label for each user's interactions
    user_split_label = []
    uid_sorted = sorted(user_inter_dict.keys())
    for uid in uid_sorted:
        interactions = user_inter_dict[uid]
        inter_len = len(interactions)
        # Using the leave-one-out strategy to split the interactions
        split_label = [0] * (inter_len - 2) + [1, 2]

        user_split_label.extend(split_label)

    inter_data['split_label'] = user_split_label

    return inter_data


def reindex_item_features(dataset_dir, save_dir, meta_file_name, iid_field):
    """
    Reindex item feature ID with IDs generated before
    """
    iid_mapping_file = os.path.join(save_dir, 'i_id_mapping.csv')
    iid_mapping = pd.read_csv(iid_mapping_file, sep=',')
    print(f'Item mapping size: {iid_mapping.shape}')

    print('3.0. Extracting User-Item Interaction Data...')
    meta_df = getDF(os.path.join(dataset_dir, meta_file_name))
    print(f'Meta data size: {meta_df.shape}')

    # 3.1. Remap item feature ID
    map_dict = dict(zip(iid_mapping['asin'], iid_mapping[iid_field]))

    meta_df[iid_field] = meta_df['asin'].map(map_dict)
    meta_df.dropna(subset=[iid_field], inplace=True)
    meta_df[iid_field] = meta_df[iid_field].astype(int)
    meta_df.sort_values(by=[iid_field], inplace=True)

    old_cols = meta_df.columns.tolist()
    new_cols = [old_cols[-1]] + old_cols[:-1]
    print(f'New columns: {new_cols}')

    meta_df = meta_df[new_cols]

    print(f'After remapping item feature ID: {meta_df.shape}')
    return meta_df


def feature_extraction(data, img_dir, save_dir, iid_field):
    """
    Extract text and image features from item data with the given model
    """

    all_iids = set(data[iid_field].unique())

    # # 4.1 Download images
    # print("Downloading images...")
    # download_images(data, img_dir)

    # 4.2 Extract text and image features
    data.sort_values(by=[iid_field], inplace=True)
    print(f'Item feature size: {data.shape}')

    # Sentences: 'title'
    title_na_df = data[data['title'].isnull()]
    print(f'Item title missing: {title_na_df.shape}')

    data['title'] = data['title'].fillna('')
    sentences = []
    for i, row in data.iterrows():
        if len(row['title']) > 1:
            sen = 'The title of this item is: {}.'.format(row['title'])
        else:
            sen = 'The title of this item is missing.'

        sentences.append(sen)

    print(f'Item title sentences: {len(sentences)}')

    course_list = data[iid_field].tolist()
    assert course_list[-1] == len(course_list) - 1

    # 4.3 Encode text and image features by CLIP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Specify the model and device, currently only supports open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)

    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer(sentences).to(device)

    image_features = {}
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
        for _, _, files in os.walk(img_dir):
            for file in tqdm(files, desc='Extracting image features', unit='images'):
                iid = int(file.split('.')[0])
                if data['asin'].isin([iid]).any():
                    idx = data.loc[data['asin'] == iid]['itemID'].iloc[0]
                    # print(f'Processing image: {iid}/ {len(files)}')
                    image_path = os.path.join(img_dir, file)
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    img_features = model.encode_image(image)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    image_features[str(idx)] = img_features.cpu().numpy()

        # Handle missing image features by using the average image features
        avg_img_features = np.mean(list(image_features.values()), axis=0)
        missing_iids = all_iids - set(map(int, image_features.keys()))

        if missing_iids:
            print(f'Missing image features: {len(missing_iids)}')
            for iid in missing_iids:
                image_features[iid] = avg_img_features

        # 将键转换为整数，并按整数顺序排序
        sorted_keys = sorted(image_features.keys(), key=int)

        # 根据排序后的键，按顺序拼接对应的行向量
        sorted_features = np.array([image_features[key] for key in sorted_keys])
        final_image_features = sorted_features.squeeze(axis=1)

        # Compute the cosine similarity between text and image features
        similarity = compute_modalities_similarity(text_features, final_image_features)

    np.save(os.path.join(save_dir, 'text_features.npy'), text_features)
    np.save(os.path.join(save_dir, 'image_features.npy'), final_image_features)
    np.save(os.path.join(save_dir, 'similarity.npy'), similarity)
