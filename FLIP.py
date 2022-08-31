import os
import re
import copy
import time
import clip
import nltk
import string
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from nltk.stem.porter import PorterStemmer
from PIL import Image
from torchvision import transforms
import timm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

img_models_params = {
    "resnet18": {
        "img_shape": 256,
        "emb_size": 512,
    },
    "vit_base_patch32_224": {
        "img_shape": 224,
        "emb_size": 768,
    },
    "tf_efficientnetv2_b3": {
        "img_shape": 300,
        "emb_size": 1536,
    },
    "mixer_b16_224": {
        "img_shape": 224,
        "emb_size": 768,
    },
}

txt_models_params = {
    "bert-base-uncased": {"emb_size": 768, "max_seq_length": 256},
    "bert-base-cased": {"emb_size": 768, "max_seq_length": 256},
    "CLIP_Transformer": {"emb_size": 1024, "max_seq_length": 77},
}

# "freeze_model" be used for fine-tuning parts of Huggingface's text models. We do not use it for CLIP's text encoder.
def freeze_model(input_model):

    count_blocks = []
    count_all_parameters = 0

    for name, param in input_model.named_parameters():

        count_all_parameters += 1

        current_param = name.split(".")[2]

        current_block = 0
        try:
            current_block = int(current_param)

            if not current_block in count_blocks:
                count_blocks.append(current_block)
        except:
            pass

    last_block = max(count_blocks)
    count_parameters = 0

    for name, param in input_model.named_parameters():

        count_parameters += 1
        param.requires_grad = False

        current_param = name.split(".")[2]

        if current_param == last_block:
            param.requires_grad = True

        if count_parameters >= count_all_parameters - 2:
            param.requires_grad = True

    return input_model


class FLIP(nn.Module):
    def __init__(
        self,
        img_models_params,
        txt_models_params,
        device,
        emb_dim=128,
        choose_image_encoder=None,
        choose_text_encoder=None,
        extract_features=False,
    ):

        super().__init__()

        self.emb_dim = emb_dim
        self.img_models_params = img_models_params
        self.txt_models_params = txt_models_params

        self.choose_image_encoder = choose_image_encoder
        self.choose_text_encoder = choose_text_encoder

        self.extract_features = extract_features

        self.image_encoder = torch.nn.Sequential(
            *(
                list(
                    timm.create_model(
                        self.choose_image_encoder, pretrained=True
                    ).children()
                )[:-1]
            )
        )

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = True

        if self.choose_text_encoder == "CLIP_Transformer":

            clip_model, _ = clip.load("RN50", device=device, jit=False)
            for name, param in clip_model.named_parameters():
                param.requires_grad = False

            self.text_encoder = clip_model.encode_text

        else:

            self.text_encoder = torch.nn.Sequential(
                *(
                    list(
                        freeze_model(
                            AutoModel.from_pretrained(self.choose_text_encoder)
                        ).children()
                    )[:-1]
                )
            )

        self.fcl_img = nn.Linear(
            self.img_models_params[self.choose_image_encoder]["emb_size"], self.emb_dim
        )

        self.fcl_txt = nn.Linear(
            self.txt_models_params[self.choose_text_encoder]["emb_size"], self.emb_dim
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):

        x_img = self.image_encoder(images)

        if self.choose_image_encoder.startswith(
            "vit"
        ) or self.choose_image_encoder.startswith("mixer"):
            x_img = torch.mean(x_img, axis=1)

        if self.choose_text_encoder == "CLIP_Transformer":
            x_txt = self.text_encoder(texts).float()

        else:
            x_txt = self.text_encoder(texts).last_hidden_state
            x_txt = torch.mean(x_txt, axis=1)

        x_img = self.fcl_img(x_img)
        x_txt = self.fcl_txt(x_txt)

        if self.extract_features:
            return x_img, x_txt

        # normalized features
        x_img = x_img / x_img.norm(dim=1, keepdim=True)
        x_txt = x_txt / x_txt.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * x_img @ x_txt.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class DatasetIterator_ImageText(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        data_path,
        transform,
        tokenizer,
        choose_text_model,
        return_ids=False,
    ):
        self.input_data = input_data
        self.return_ids = return_ids
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.choose_text_model = choose_text_model

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        item_id = current.item_id

        # IMAGES
        image_path = self.data_path + "images/" + item_id + ".jpg"
        img = Image.open(image_path)
        img = self.transform(img)

        # TEXTS
        if self.choose_text_model == "CLIP_Transformer":
            try:
                txt = torch.squeeze(clip.tokenize(current.text[:77]))
            except:
                txt = torch.squeeze(clip.tokenize(""))
        else:
            txt = np.array(
                self.tokenizer(current.text, padding="max_length", truncation=True)[
                    "input_ids"
                ]
            )

        if self.return_ids:
            return img, txt, item_id
        else:
            return img, txt


def train_flip(
    choose_image_model,
    EPOCHS=20,
    LEARNING_RATE=1e-4,
    emb_size=512,
    choose_gpu=0,
    memory_fraction=1.0,
    batch_size=32,
    num_workers=8,
    early_stop_epochs=10,
    data_path="data_benchmark/polyvore/polyvore_outfits/",
    polyvore_version="disjoint",
    choose_text_model="CLIP_Transformer",
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=choose_gpu)

    def save_results_csv(output_folder_, output_file_, model_performance_):
        print("Save Results ", end=" ... ")
        exp_results_pd = pd.DataFrame(pd.Series(model_performance_)).transpose()
        if not os.path.isfile(output_folder_ + "/" + output_file_ + ".csv"):
            exp_results_pd.to_csv(
                output_folder_ + "/" + output_file_ + ".csv",
                header=True,
                index=False,
                columns=list(model_performance_.keys()),
            )
        else:
            exp_results_pd.to_csv(
                output_folder_ + "/" + output_file_ + ".csv",
                mode="a",
                header=False,
                index=False,
                columns=list(model_performance_.keys()),
            )
        print("Done\n")

    img_shape = img_models_params[choose_image_model]["img_shape"]
    transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if choose_text_model == "CLIP_Transformer":
        tokenizer = clip.tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(choose_text_model)

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

        return item_text

    def preprocess_text(text):
        punctuationfree = "".join(
            [i for i in text if i not in punctuation and not i.isdigit()]
        )
        tokens = re.split("W+", punctuationfree)
        remove_stopwords = [i.lower() for i in tokens if i not in stopwords]
        stem_text = [porter_stemmer.stem(word) for word in remove_stopwords]
        output = " ".join(stem_text)
        return output

    def remap_dataset_FLIP(input_data):
        category_x_itemID = (
            item_metadata.transpose().reset_index().rename({"index": "item_id"}, axis=1)
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

                garment_dict = {
                    "item_id": item,
                    "category_id": category_id,
                    "category_name": categories[int(category_id)],
                    "text": preprocess_text(fetch_item_text(item)),
                }

                remaped_data.append(garment_dict)

        return pd.DataFrame(remaped_data).drop_duplicates("item_id")

    if not os.path.isfile(
        data_path + "misfits/" + "FLIP_" + polyvore_version + "_train.pkl"
    ):

        print("Fetch and remap Polyvore")
        punctuation = string.punctuation
        stopwords = nltk.corpus.stopwords.words("english")
        porter_stemmer = PorterStemmer()

        print("Read category file")
        categories = pd.read_csv(
            data_path + "categories.csv", header=None, index_col=None
        )
        categories.columns = ["category_id", "category", "high_level"]
        categories.index = categories.category_id
        categories = categories["category"]
        categories = categories.to_dict()

        print("Read Item Metadata")
        f = open(data_path + "polyvore_item_metadata.json", "r")
        item_metadata = json.loads(f.read())
        item_metadata = pd.DataFrame(item_metadata)

        train_df = remap_dataset_FLIP(input_data="train")
        valid_df = remap_dataset_FLIP(input_data="valid")
        test_df = remap_dataset_FLIP(input_data="test")

        train_df.to_pickle(
            data_path + "misfits/" + "FLIP_" + polyvore_version + "_train.pkl"
        )
        valid_df.to_pickle(
            data_path + "misfits/" + "FLIP_" + polyvore_version + "_valid.pkl"
        )
        test_df.to_pickle(
            data_path + "misfits/" + "FLIP_" + polyvore_version + "_test.pkl"
        )

    else:
        print("Load re-mapped Polyvore")
        train_df = pd.read_pickle(
            data_path + "misfits/" + "FLIP_" + polyvore_version + "_train.pkl"
        )
        valid_df = pd.read_pickle(
            data_path + "misfits/" + "FLIP_" + polyvore_version + "_valid.pkl"
        )
        test_df = pd.read_pickle(
            data_path + "misfits/" + "FLIP_" + polyvore_version + "_test.pkl"
        )

    train_dg = DatasetIterator_ImageText(
        train_df,
        data_path=data_path,
        transform=transform,
        tokenizer=tokenizer,
        choose_text_model=choose_text_model,
    )

    valid_dg = DatasetIterator_ImageText(
        valid_df,
        data_path=data_path,
        transform=transform,
        tokenizer=tokenizer,
        choose_text_model=choose_text_model,
    )

    test_dg = DatasetIterator_ImageText(
        test_df,
        data_path=data_path,
        transform=transform,
        tokenizer=tokenizer,
        choose_text_model=choose_text_model,
    )

    train_dataloader = DataLoader(
        train_dg,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dg,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dg,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = FLIP(
        img_models_params=img_models_params,
        txt_models_params=txt_models_params,
        emb_dim=emb_size,
        device=device,
        choose_image_encoder=choose_image_model,
        choose_text_encoder=choose_text_model,
        extract_features=False,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.001,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1, verbose=True
    )

    def model_step():
        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), device=device)
        images_loss = criterion(logits_per_image, ground_truth)
        text_loss = criterion(logits_per_text, ground_truth)

        loss = (images_loss + text_loss) / 2

        return loss

    history = []
    has_not_improved_for = 0

    PATH = (
        "checkpoints_pt/FLIP_"
        + choose_image_model
        + "_"
        + choose_text_model
        + "_"
        + str(emb_size)
        + "_"
        + str(LEARNING_RATE)
        + "_"
        + str(batch_size)
        + ".pt"
    )

    for epoch in range(EPOCHS):

        epoch_start_time = time.time()

        running_loss = 0.0

        model.train()
        batches_per_epoch = train_dg.__len__() // batch_size

        for i, data in enumerate(train_dataloader, 0):

            images, texts = (
                data[0].to(device, non_blocking=True),
                data[1].to(device, non_blocking=True),
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model_step()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(
                f"[Epoch:{epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}",
                end="\r",
            )

        print("\nEvaluation:", end=" -> ")

        model.eval()
        val_running_loss = 0

        with torch.no_grad():
            for j, data in enumerate(valid_dataloader, 0):

                images, texts = (
                    data[0].to(device, non_blocking=True),
                    data[1].to(device, non_blocking=True),
                )

                val_loss = model_step()
                val_running_loss += val_loss

        validation_loss = val_running_loss / (j + 1)
        validation_loss = validation_loss.cpu().detach().numpy()

        print("Validation loss:", validation_loss)

        history.append(validation_loss)

        print("Checkpoint?", end="...")
        if epoch == np.argmin(np.array(history)):
            # checkpoint
            print("Checkpoint!!!\n")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                PATH,
            )
            has_not_improved_for = 0
        else:
            has_not_improved_for += 1

        if has_not_improved_for > early_stop_epochs:
            break

        scheduler.step()

        print("\n")

    print("Finished Training. Loading the best model from checkpoints.")

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    model.eval()
    test_running_loss = 0

    batches_per_epoch = test_dg.__len__() // batch_size

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):

            print("Batch:", i, "out of", batches_per_epoch, end="\r")

            images, texts = (
                data[0].to(device, non_blocking=True),
                data[1].to(device, non_blocking=True),
            )

            test_loss = model_step()
            test_running_loss += test_loss

    testing_loss = test_running_loss / (i + 1)

    result = {}
    result["choose_image_model"] = choose_image_model
    result["choose_text_model"] = choose_text_model
    result["test_loss"] = testing_loss.detach().cpu().numpy().tolist()
    result["emb_size"] = emb_size
    result["batch_size"] = batch_size
    result["learning_rate"] = LEARNING_RATE
    result["history"] = history
    result["checkpoint_path"] = PATH
    result["NOTES"] = ""

    save_results_csv(
        "data_benchmark/polyvore/polyvore_outfits/results/",
        "RESULTS_FLIP",
        result,
    )


def extract_visual_features(
    data_path,
    choose_image_model,
    choose_text_model,
    pretraining_method,
    batch_size=64,
    polyvore_version="nondisjoint",
    choose_gpu=0,
    FLIP_learning_rate=1e-4,
    FLIP_emb_size=512,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)

    img_shape = img_models_params[choose_image_model]["img_shape"]
    transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if choose_text_model == "CLIP_Transformer":
        tokenizer = clip.tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(choose_text_model)

    print("Load re-mapped Polyvore")
    train_df = pd.read_pickle(
        data_path + "misfits/" + "FLIP_" + "nondisjoint" + "_train.pkl"
    )
    valid_df = pd.read_pickle(
        data_path + "misfits/" + "FLIP_" + "nondisjoint" + "_valid.pkl"
    )
    test_df = pd.read_pickle(
        data_path + "misfits/" + "FLIP_" + "nondisjoint" + "_test.pkl"
    )

    all_data = pd.concat([train_df, valid_df, test_df])
    all_data = all_data.drop_duplicates("item_id")

    all_data_dg = DatasetIterator_ImageText(
        all_data,
        transform=transform,
        tokenizer=tokenizer,
        data_path=data_path,
        choose_text_model=choose_text_model,
        return_ids=True,
    )

    dataloader = DataLoader(
        all_data_dg,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    flip_model = FLIP(
        img_models_params=img_models_params,
        txt_models_params=txt_models_params,
        emb_dim=FLIP_emb_size,
        device=device,
        choose_image_encoder=choose_image_model,
        choose_text_encoder=choose_text_model,
        extract_features=True,
    )

    flip_model.to(device)

    if pretraining_method == "FLIP":
        PATH = (
            "checkpoints_pt/FLIP_"
            + choose_image_model
            + "_"
            + choose_text_model
            + "_"
            + str(FLIP_emb_size)
            + "_"
            + str(FLIP_learning_rate)
            + "_"
            + str(32)
            + ".pt"
        )

        checkpoint = torch.load(PATH)
        flip_model.load_state_dict(checkpoint["model_state_dict"])

    all_vf = []
    all_tf = []
    all_ids = []

    flip_model.eval()

    all_batches = all_data_dg.__len__() // batch_size

    with torch.no_grad():

        for i, data in enumerate(dataloader):

            print(i + 1, "batch out of", all_batches, end="\r")

            imgs, txts, item_ids = (
                data[0].to(device, non_blocking=True),
                data[1].to(device, non_blocking=True),
                data[2],
            )

            img_features, txt_features = flip_model(imgs, txts)

            img_features = img_features.cpu().detach().numpy()
            txt_features = txt_features.cpu().detach().numpy()

            all_vf.extend(img_features)
            all_tf.extend(txt_features)
            all_ids.extend(item_ids)

    all_vf = np.stack(all_vf)
    all_tf = np.stack(all_tf)

    np.save(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_vf.npy",
        all_vf,
    )
    np.save(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_tf.npy",
        all_tf,
    )

    with open(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_item_ids",
        "w",
    ) as fp:
        json.dump(all_ids, fp)