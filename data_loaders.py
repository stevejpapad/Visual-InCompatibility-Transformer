import clip
import torch 
import numpy as np
from PIL import Image

class DatasetIterator_VICTOR(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        data_path,
        vf_df,
        tf_df,
        vf_shape,
        tf_shape,
        limit_items=10,
        use_misfits=False,
        use_features=["images"],
        fine_tune_vf=False,
    ):
        self.input_data = input_data
        self.use_misfits = use_misfits
        self.use_features = use_features
        self.limit_items = limit_items
        self.fine_tune_vf = fine_tune_vf
        self.data_path = data_path
        self.vf_df = vf_df
        self.tf_df = tf_df
        self.vf_shape = vf_shape
        self.tf_shape = tf_shape

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]

        pad0 = current["PAD0_N"].astype("float32")
        current_items = current.outfit_items
        current_items = current_items[current_items != "0"]

        if "images" in self.use_features:
            if self.fine_tune_vf:

                images = []
                for item_id in current_items:
                    image_path = data_path + "images/" + item_id + ".jpg"
                    img = Image.open(image_path)
                    img = transform(img)
                    img = img.reshape(1, 3, img_shape, img_shape)
                    images.append(img)

                images = torch.vstack(images)
                pad_zeros = np.zeros(
                    (self.limit_items - images.shape[0], 3, img_shape, img_shape)
                )
                x = np.vstack([pad_zeros, images]).astype("float32")

            else:
                current_vf = self.vf_df[current_items].transpose().values
                pad_zeros = np.zeros(
                    (self.limit_items - current_items.shape[0], self.vf_shape)
                )
                x = np.vstack([pad_zeros, current_vf]).astype("float32")

        if "texts" in self.use_features:
            current_tf = self.tf_df[current_items].transpose().values
            pad_zeros = np.zeros(
                (self.limit_items - current_items.shape[0], self.tf_shape)
            )
            x_tf = np.vstack([pad_zeros, current_tf]).astype("float32")

            if "images" in self.use_features and not self.fine_tune_vf:
                x = np.concatenate([x, x_tf], axis=1)
            elif "images" in self.use_features and self.fine_tune_vf:
                # TO-DO
                pass
            else:
                x = x_tf

        y = current["target"].astype("float32")

        if self.use_misfits:
            y2 = current["MID_target"].astype("float32")
            return x, (y, y2), pad0

        return x, y, pad0

class DatasetIterator_FLIP(torch.utils.data.Dataset):
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

