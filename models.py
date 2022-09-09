import timm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import clip

class VICTOR(nn.Module):
    def __init__(
        self,
        device,
        emb_dim=64,
        tf_layers=1,
        tf_head=2,
        tf_dim=128,
        activation="relu",
        dropout=0.1,
        limit_items=19,
        use_misfits=False,
        use_OCr=False,
        use_MID=False,
        fine_tune_vf=False,
        choose_cv_model=None,
        use_cls_token=False,
        use_extra_attention=False,
    ):

        super().__init__()

        self.use_misfits = use_misfits
        self.emb_dim = emb_dim
        self.use_OCr = use_OCr
        self.use_MID = use_MID
        self.fine_tune_vf = fine_tune_vf
        self.choose_cv_model = choose_cv_model
        self.use_cls_token = use_cls_token
        self.use_extra_attention = use_extra_attention

        if self.fine_tune_vf:

            self.cv_model = nn.Sequential(
                *(
                    list(
                        timm.create_model(
                            self.choose_cv_model, pretrained=True
                        ).children()
                    )[:-1]
                )
            )
            self.img_shape = cv_models_params[self.choose_cv_model]["img_shape"]

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=tf_layers,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.limit_items = limit_items

        if self.use_cls_token:
            self.class_token = nn.Parameter(
                nn.init.zeros_(torch.empty(1, 1, self.emb_dim)), requires_grad=True
            )

        self.fcl = nn.Linear(self.emb_dim, self.emb_dim // 2)

        if not self.use_misfits or self.use_OCr:
            self.output_score = nn.Linear(self.emb_dim // 2, 1)

        if self.use_misfits and use_MID:
            self.output_mid_list = nn.ModuleList()
            self.layer_norms_mid = nn.ModuleList()

            for i in range(0, self.limit_items):
                self.layer_norms_mid.append(nn.LayerNorm(self.emb_dim).to(device))
                self.output_mid_list.append(nn.Linear(self.emb_dim, 1).to(device))

    def forward(self, x, pad):
        b_size = x.shape[0]

        if self.fine_tune_vf:

            x = x.reshape(-1, 3, self.img_shape, self.img_shape)
            x = self.cv_model(x)
            x = x.reshape(b_size, self.limit_items, -1)

        if self.use_cls_token:
            cls_token = torch.broadcast_to(
                self.class_token, (b_size, 1, self.emb_dim)
            )
            x = torch.cat([cls_token, x], axis=1)

        x = self.transformer(x)

        if self.use_cls_token:
            x1 = x[:, 0, :]
            x2 = x[:, 1:, :]
        else:
            x = x.mean(1)

        if not self.use_misfits or self.use_OCr:
            x1 = self.layer_norm(x1)
            x1 = self.dropout(x1)
            x1 = self.fcl(x1)
            x1 = self.gelu(x1)
            x1 = self.dropout(x1)
            y1 = self.output_score(x1)

        if self.use_MID:

            y2 = []

            for i in range(self.limit_items):

                x2_current = x2[:, i, :]
                x2_current = self.layer_norms_mid[i](x2_current)
                x2_current = self.dropout(x2_current)
                x2_current = self.gelu(x2_current)

                y2.append(self.output_mid_list[i](x2_current))

            y2 = torch.stack(y2, dim=1).squeeze()

            if not self.use_OCr:
                return None, y2

        return y1, y2

        
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

        self.image_encoder = nn.Sequential(
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

            self.text_encoder = nn.Sequential(
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