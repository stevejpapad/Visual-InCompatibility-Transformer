import re
import nltk
import string
from nltk.stem.porter import PorterStemmer
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import copy
from torch.utils.data import DataLoader

random.seed(0)
np.random.seed(0)


def fetch_polyvore(
    data_path,
    use_misfits,
    generate_per_outfit,
    limit_items,
    use_descriptions,
    polyvore_version,
):

    punctuation = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")
    porter_stemmer = PorterStemmer()

    print("Read category file")
    categories = pd.read_csv(data_path + "categories.csv", header=None, index_col=None)
    categories.columns = ["category_id", "category", "high_level"]
    categories.index = categories.category_id
    categories = categories["category"]
    categories = categories.to_dict()

    print("Read Item Metadata")
    f = open(data_path + "polyvore_item_metadata.json", "r")
    item_metadata = json.loads(f.read())
    item_metadata = pd.DataFrame(item_metadata)

    # Encode item IDs and categories
    unique_item_ids = np.array(item_metadata.columns)
    unique_categories = item_metadata.transpose().category_id.unique().tolist()

    def preprocess_text(text):
        punctuationfree = "".join(
            [i for i in text if i not in punctuation and not i.isdigit()]
        )
        tokens = re.split("W+", punctuationfree)
        remove_stopwords = [i.lower() for i in tokens if i not in stopwords]
        stem_text = [porter_stemmer.stem(word) for word in remove_stopwords]

        output = " ".join(stem_text)

        return output

    def limit_n_pad_items(arr, n):
        arr = np.array(arr[:n])
        arr = np.pad(arr, (n - arr.shape[0], 0), "constant", constant_values=(0))
        return arr

    def fetch_item_text(item_id, use_desc=False):

        url_name = item_metadata[item_id]["url_name"]
        title_text = item_metadata[item_id]["title"]
        desc_text = item_metadata[item_id]["description"]

        item_text = ""
        if title_text != "":
            item_text += title_text

        else:
            item_text = url_name

        if use_desc:
            item_text += ". " + desc_text

        item_text = preprocess_text(item_text)

        return item_text  # , vf

    def choose_random_w_exception(lst, exception):

        tries = 0
        while True:
            choice = random.choice(lst)
            if choice != exception:
                return choice

            tries += 1

            if tries > 4:
                return "0"

    def n_to_generate(generate_per_outfit, current_outfit):

        to_change_list = []
        outfit_scores = []

        for _ in range(generate_per_outfit):

            non_zero_items = current_outfit[current_outfit != "0"]
            items_in_outfit = non_zero_items.shape[0]

            max_to_change = items_in_outfit - 2
            how_many_to_change = np.random.randint(1, max_to_change + 1)

            change_these = np.zeros(items_in_outfit)
            change_these[:how_many_to_change] = 1.0
            np.random.shuffle(change_these)

            actual_item_positions = non_zero_items[np.nonzero(change_these)[0]]
            new_outfit_score = round(
                1 - how_many_to_change / non_zero_items.shape[0], 4
            )

            to_change_list.append(actual_item_positions)
            outfit_scores.append(new_outfit_score)

        return to_change_list, outfit_scores

    def remap_data(
        input_data,
        polyvore_version="nondisjoint",
        limit_items=19,
        use_desc=False,
        generate_per_outfit=0,
    ):

        with open(data_path + "item_ids", "r") as fp:
            cols = json.load(fp)

        category_x_itemID = (
            item_metadata[cols]
            .transpose()
            .reset_index()
            .rename({"index": "item_id"}, axis=1)
        )
        category_x_itemID = category_x_itemID.groupby("category_id")["item_id"].apply(
            list
        )

        with open(
            data_path + polyvore_version + "/compatibility_" + input_data + ".txt"
        ) as f:
            lines = f.readlines()

        f = open(data_path + polyvore_version + "/" + input_data + ".json", "r")
        data = json.loads(f.read())
        df = pd.DataFrame(data)
        df.index = df.set_id
        df = df["items"]

        remaped_data = []

        for line in tqdm(lines, total=len(lines)):
            split_line = line.split()

            target = int(split_line[0])
            outfits_plus_idx = split_line[1:]

            outfit_idxs = (
                [x.split("_")[0] for x in outfits_plus_idx]
                if target == 0
                else [outfits_plus_idx[0].split("_")[0]]
            )
            item_idxs = [x.split("_")[1] for x in outfits_plus_idx]

            count = 0
            (
                actual_items,
                item_categories,
                item_categories_names,
                item_texts,
                visual_features,
            ) = ([], [], [], [], [])

            if len(outfit_idxs) == 1:
                actual_items = [x["item_id"] for x in df[outfit_idxs[0]]]
            else:
                current_outfit_df = df[outfit_idxs]
                for i in range(len(item_idxs)):
                    current_item = current_outfit_df[i][int(item_idxs[i]) - 1][
                        "item_id"
                    ]
                    actual_items.append(current_item)

            for item in actual_items:
                category_id = item_metadata[item]["category_id"]
                item_categories.append(category_id)
                item_categories_names.append(categories[int(category_id)])

            outfit_dict = {
                "target": target,
                "outfit_items": limit_n_pad_items(actual_items, limit_items).astype(
                    "U16"
                ),
                "item_categories": limit_n_pad_items(item_categories, limit_items),
                "category_names": item_categories_names[:limit_items],
                "MID_target": np.zeros(limit_items)
                if target == 1
                else np.ones(limit_items),
                "altered": False,
                "PAD0_N": limit_items - len(actual_items),
            }

            remaped_data.append(outfit_dict)

            count_nonzero = outfit_dict["outfit_items"][
                outfit_dict["outfit_items"] != "0"
            ].shape[0]

            if target == 1 and generate_per_outfit > 0 and count_nonzero > 2:

                for _ in range(generate_per_outfit):

                    to_change_list, outfit_scores = n_to_generate(
                        1, outfit_dict["outfit_items"]
                    )

                    for i in range(len(to_change_list)):

                        mid_array = np.zeros(limit_items)
                        copy_outfit = copy.deepcopy(outfit_dict)

                        positions_items_to_change = []

                        for x in to_change_list[0]:
                            positions_items_to_change.extend(
                                np.where(outfit_dict["outfit_items"] == x)[0]
                            )

                        for j in positions_items_to_change:

                            current_cat = outfit_dict["item_categories"][j]
                            current_item_id = outfit_dict["outfit_items"][j]

                            candidates = category_x_itemID[str(current_cat)]
                            substitute_item = choose_random_w_exception(
                                candidates, current_item_id
                            )

                            # category_id, item_text, vf
                            item_text = fetch_item_text(
                                substitute_item, use_desc=use_desc
                            )

                            copy_outfit["outfit_items"][j] = substitute_item
                            mid_array[j] = 1

                        copy_outfit["MID_target"] = mid_array
                        copy_outfit["target"] = outfit_scores[i]
                        copy_outfit["altered"] = True

                        remaped_data.append(copy_outfit)

        return pd.DataFrame(remaped_data)

    if use_misfits:
        to_save_path = (
            data_path
            + "misfits/"
            + polyvore_version
            + "_gen"
            + str(generate_per_outfit)
            + "_"
            + str(limit_items)
        )

        if not os.path.isfile(to_save_path + "_train.pkl"):
            print("Generate outfits")

            train_df = remap_data(
                input_data="train",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=generate_per_outfit,
            )
            train_df = train_df.sample(frac=1).reset_index(drop=True)
            valid_df = remap_data(
                input_data="valid",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=generate_per_outfit,
            )
            test_df = remap_data(
                input_data="test",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=generate_per_outfit,
            )
            original_test_df = remap_data(
                input_data="test",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=0,
            )

            if not os.path.isdir(data_path + "misfits"):
                os.mkdir(data_path + "misfits")

            train_df.to_pickle(to_save_path + "_train.pkl")
            valid_df.to_pickle(to_save_path + "_valid.pkl")
            test_df.to_pickle(to_save_path + "_test.pkl")

        else:

            print("Load Misfit dataset")
            train_df = pd.read_pickle(to_save_path + "_train.pkl")
            valid_df = pd.read_pickle(to_save_path + "_valid.pkl")
            test_df = pd.read_pickle(to_save_path + "_test.pkl")

    else:

        to_save_path = (
            data_path
            + "misfits/"
            + polyvore_version
            + "_default"
            + "_"
            + str(limit_items)
        )

        if not os.path.isfile(to_save_path + "_train.pkl"):
            print("Generate outfits")

            print("Remap Polyvore")
            train_df = remap_data(
                input_data="train",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=0,
            )

            train_df = train_df.sample(frac=1).reset_index(drop=True)

            valid_df = remap_data(
                input_data="valid",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=0,
            )

            test_df = remap_data(
                input_data="test",
                polyvore_version=polyvore_version,
                limit_items=limit_items,
                use_desc=use_descriptions,
                generate_per_outfit=0,
            )

            if not os.path.isdir(data_path + "misfits"):
                os.mkdir(data_path + "misfits")

            train_df.to_pickle(to_save_path + "_train.pkl")
            valid_df.to_pickle(to_save_path + "_valid.pkl")
            test_df.to_pickle(to_save_path + "_test.pkl")

        else:
            print("Load polyvore dataset")
            train_df = pd.read_pickle(to_save_path + "_train.pkl")
            valid_df = pd.read_pickle(to_save_path + "_valid.pkl")
            test_df = pd.read_pickle(to_save_path + "_test.pkl")

    return train_df, valid_df, test_df


def fetch_features(data_path, choose_image_model, pretraining_method):

    print("Load visual features")
    with open(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_item_ids",
        "r",
    ) as fp:
        cols = json.load(fp)

    vf = np.load(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_vf.npy"
    )
    vf_df = pd.DataFrame(vf.T, columns=cols)

    print("Load textual features")
    tf = np.load(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_tf.npy"
    )

    tf_df = pd.DataFrame(tf.T, columns=cols)

    return vf_df, tf_df