import os

from utils.utils import read_yaml_to_dict
from utils.data import rating2inter, loo_split, reindex_item_features, feature_extraction, gen_user_graph


def quick_start(config, dataset_name):
    dataset_folder = config['dataset'][dataset_name]
    dataset_config = read_yaml_to_dict(os.path.join(dataset_base, dataset_folder, 'config.yaml'))

    rating_file = dataset_config['file']['rating']
    meta_file = dataset_config['file']['item']
    img_folder = dataset_config['folder']['image']

    user_id, item_id, rating, timestamp, min_u_num, min_i_num, ratios = (
        dataset_config['field']['user_id'], dataset_config['field']['item_id'],
        dataset_config['field']['rating'], dataset_config['field']['timestamp'],
        config['config']['min_user_interactions'], config['config']['min_item_interactions'],
        config['config']['split_ratio']
    )

    # 0.1. Create save path and image folder
    dataset_path = os.path.join(dataset_base, dataset_folder)
    save_path = os.path.join(data_save_dir, dataset_folder)
    image_path = os.path.join(dataset_path, img_folder)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(image_path):
        # Create image folder
        os.makedirs(image_path)

    # 1. Convert rating data to interaction data and split it into training, validation, and test sets
    df = rating2inter(user_id, item_id, rating, timestamp, dataset_path, save_path, rating_file,
                      min_u_num, min_i_num, ratios)

    # 2. Leave-one-out splitting
    df = loo_split(df, user_id, item_id, timestamp)

    inter_file = os.path.join(save_path, 'inter.csv')
    df.to_csv(inter_file, sep=',', index=False)
    print('The interaction data is saved!')

    # # 3. Reindex item feature ID with IDs generated before
    # item_df = reindex_item_features(dataset_path, save_path, meta_file, item_id)
    #
    # # 4. Text & Image feature extraction
    # feature_extraction(item_df, image_path, save_path, item_id)

    # 5. Generate user-user graph matrix
    print(f'Generating u-u matrix for {dataset_name} ...\n')
    gen_user_graph(df, user_id, item_id, save_path)
    print('The user-user graph matrix is saved!')





if __name__ == '__main__':

    datasets = ['Amazon_Baby', 'Bili_Food', 'Bili_Movie', 'Bili_Dance', 'KU', 'DY']
    # datasets = ['Bili_Food', 'Bili_Movie', 'Bili_Dance']
    # datasets = ['KU']


    dataset_base = './datasets/'
    data_save_dir = './processed_datasets/'

    # 0. Load dataset config
    args = read_yaml_to_dict(os.path.join(dataset_base, 'general.yaml'))

    for dataset in datasets:
        print(f'>>>> Processing the {dataset} dataset... <<<<')

        try:
            quick_start(args, dataset)
        except Exception as e:
            print(f'Error: {e}')
            continue