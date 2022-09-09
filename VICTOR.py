import timm
import time
import os
import json
import numpy as np
from sklearn import metrics
import pandas as pd
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from prepare_polyvore import fetch_features, fetch_polyvore
from torchvision import models, transforms
from data_loaders import DatasetIterator_VICTOR
from models import VICTOR
import gc


def run_experiment(
    choose_gpu=0,
    data_path="data_benchmark/polyvore/polyvore_outfits/",
    memory_fraction=1.0,
    batch_size=512,
    use_MID_choices=[False, True],
    use_OCr_choices=[False, True],
    dropout_choices=[0.2],
    tf_layers_choices=[8],
    tf_head_choices=[16],
    tf_dim_choices=[64],
    w_loss_choices=[1],  # 'a' in paper
    notes="",
    save_results_to="results_MISFITS",
    use_misfits=True,
    use_cls_token=True,
    pretraining_method="FLIP",
    polyvore_version="nondisjoint",
    choose_image_model="resnet18",
    FLIP_emb_size=512,
    FLIP_learning_rate=1e-4,
    choose_text_model="CLIP_Transformer",
    fine_tune_vf=False,
    use_extra_attention=False,
    use_features=["images"],
    generate_per_outfit=2,  # 'm' in paper
    limit_items=19,  # uses all items in an outfit. 19 is the max items in polyvore.
    use_descriptions=True,
    num_workers=8,
    early_stop_epochs=10,
    EPOCHS=30,
    LEARNING_RATE=1e-4,
    save_results=True,
):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    gc.collect()
    generate_per_outfit = generate_per_outfit if use_misfits else 0

    batch_size = batch_size if not fine_tune_vf else 32

    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)

    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

    # extracting, saving and fetching the visual and text features
    if not os.path.isfile(
        data_path
        + "features/"
        + choose_image_model
        + "_"
        + pretraining_method
        + "_vf.npy"
    ):

        from FLIP import extract_visual_features

        extract_visual_features(
            data_path,
            choose_image_model,
            choose_text_model,
            pretraining_method,
            polyvore_version="nondisjoint",  # we use the nondisjoint dataset for feature extraction because it contains all items. But use the disjoint for training FLIP so as to avoid overlaps in training and validation
            FLIP_learning_rate=FLIP_learning_rate,
            choose_gpu=choose_gpu,
        )

    vf_df, tf_df = fetch_features(data_path, choose_image_model, pretraining_method)

    # creating, saving and fetching the polyvore dataset
    train_df, valid_df, test_df = fetch_polyvore(
        data_path=data_path,
        use_misfits=use_misfits,
        generate_per_outfit=generate_per_outfit,
        limit_items=limit_items,
        use_descriptions=use_descriptions,
        polyvore_version=polyvore_version,
    )

    # used for evaluating OCb for models trained as OCr or MID or both
    if use_misfits:
        original_test_df = test_df[test_df.altered == False].reset_index(drop=True)

    if not fine_tune_vf:
        vf_shape = vf_df.shape[0]
        tf_shape = tf_df.shape[0]
        mm_shape = vf_shape + tf_shape

    # helper and evaluation functions
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

    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def evaluate_cp(input_dataloader):
        y_pred_list = []
        y_true_list = []

        model.eval()

        with torch.no_grad():

            for i, data in enumerate(input_dataloader, 0):

                if use_misfits:
                    (
                        inputs,
                        labels,  # labels_regression,
                        labels_mid,
                        pad_positions,
                    ) = (
                        data[0].to(device, non_blocking=True),
                        data[1][0].to(device, non_blocking=True),
                        data[1][1].to(device, non_blocking=True),
                        data[2].to(device, non_blocking=True),
                    )
                else:
                    inputs, labels, pad_positions = (
                        data[0].to(device, non_blocking=True),
                        data[1].to(device, non_blocking=True),
                        data[2].to(device, non_blocking=True),
                    )

                y_pred = model(inputs, pad_positions)

                if not use_misfits:
                    y_pred = torch.sigmoid(y_pred).cpu()
                    y_pred = np.clip(y_pred, 0, 1)

                elif use_OCr:  # and not use_MID:
                    y_pred = torch.sigmoid(y_pred[0]).cpu()
                    y_pred = np.clip(y_pred, 0, 1)

                elif use_MID:
                    y_pred = torch.sigmoid(y_pred[1]).cpu()
                    y_pred = np.clip(y_pred, 0, 1)

                else:
                    raise Exception(
                        "One of use_MID, use_OCr, CPb training should be True"
                    )

                y_pred_list.extend(y_pred.numpy())
                y_true_list.extend(labels.cpu().numpy())

        y_true_list = np.array([a.squeeze() for a in y_true_list])
        y_pred_list = [a.squeeze() for a in y_pred_list]
        y_pred_list = np.array(y_pred_list)

        if use_MID and not use_OCr:
            items_per_outfit = y_pred_list.shape[1]
            y_pred_list = np.sum(1 - y_pred_list, axis=1) / items_per_outfit

        correct_results_sum = (np.round(y_pred_list) == y_true_list).sum()

        acc = correct_results_sum / y_true_list.shape[0]
        acc = np.round(acc * 100)

        print(
            "Accuracy:",
            acc,
            "AUC:",
            round(metrics.roc_auc_score(y_true_list, y_pred_list), 4),
        )
        return {"AUC": round(metrics.roc_auc_score(y_true_list, y_pred_list), 4)}

    def evaluate_CP_MID(input_dataloader):
        y_pred_reg_l, y_true_reg = [], []
        y_pred_mid_l, y_true_mid = [], []
        pad_list = []

        model.eval()

        with torch.no_grad():

            for i, data in enumerate(input_dataloader, 0):

                (inputs, labels_reg, labels_mid, pad_positions,) = (
                    data[0].to(device, non_blocking=True),
                    data[1][0].to(device, non_blocking=True),
                    data[1][1].to(device, non_blocking=True),
                    data[2].to(device, non_blocking=True),
                )

                predictions = model(inputs, pad_positions)
                pad_list.extend(pad_positions.cpu().detach().numpy())

                if use_MID:
                    y_pred_mid = torch.sigmoid(predictions[1]).cpu()
                    y_pred_mid = np.round(y_pred_mid.numpy())
                    y_pred_mid_l.extend(y_pred_mid)
                    y_true_mid.extend(labels_mid.cpu().numpy())

                if use_OCr:
                    y_pred_reg = predictions[0].cpu()
                    y_pred_reg_l.extend(y_pred_reg.numpy())
                    y_true_reg.extend(labels_reg.cpu().numpy())

        mae, mse, hamming_loss = 1.0, 1.0, 1.0
        accuracy, exact_match = 0.0, 0.0

        if use_OCr:
            y_true_reg = np.array([a.squeeze() for a in y_true_reg])
            y_pred_reg_l = [a.squeeze() for a in y_pred_reg_l]
            y_pred_reg_l = np.array(y_pred_reg_l)
            mae = metrics.mean_absolute_error(y_true_reg, y_pred_reg_l)
            mse = metrics.mean_squared_error(y_true_reg, y_pred_reg_l)

        keep_exact_matches = []

        if use_MID:

            y_pred_mid_l = np.stack(y_pred_mid_l)
            y_true_mid = np.stack(y_true_mid)
            pad_list = np.stack(pad_list)

            count_non_pad = 0

            for i in tqdm(range(y_true_mid.shape[0]), total=y_true_mid.shape[0]):

                pad0 = int(pad_list[i])
                y_pred = y_pred_mid_l[i][pad0:]
                y_true = y_true_mid[i][pad0:]

                truth_values = y_pred == y_true

                if all(truth_values):
                    exact_match += 1
                    keep_exact_matches.append(i)

                count_non_pad += y_true.shape[0]
                accuracy += np.sum(truth_values)

            accuracy = accuracy / count_non_pad
            exact_match = exact_match / y_true_mid.shape[0]

        print(
            "MAE:",
            round(mae, 4),
            "MSE:",
            round(mse, 4),
            "exact_match:",
            round(exact_match, 4),
            "hamming:",
            hamming_loss,
            "accuracy: ",
            round(accuracy, 4),
        )

        return {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "exact_match": exact_match,
            "hamming:": hamming_loss,
            "accuracy": accuracy,
        }

    def topsis(xM, wV=None):
        m, n = xM.shape

        if wV is None:
            wV = np.ones((1, n)) / n
        else:
            wV = wV / np.sum(wV)

        normal = np.sqrt(np.sum(xM**2, axis=0))

        rM = xM / normal
        tM = rM * wV
        twV = np.max(tM, axis=0)
        tbV = np.min(tM, axis=0)
        dwV = np.sqrt(np.sum((tM - twV) ** 2, axis=1))
        dbV = np.sqrt(np.sum((tM - tbV) ** 2, axis=1))
        swV = dwV / (dwV + dbV)

        arg_sw = np.argsort(swV)[::-1]

        r_sw = swV[arg_sw]

        return np.argsort(swV)[::-1]

    def choose_best_model(input_df, metrics=["CP_AUC"]):

        X0 = input_df.copy()
        X0 = X0.reset_index(drop=True)
        X1 = X0[metrics]
        X1 = X1.reset_index(drop=True)

        if "CP_AUC" in metrics:
            X1["CP_AUC"] = 1 - X1["CP_AUC"]
        if "MAE" in metrics:
            pass
        if "accuracy" in metrics:
            X1["accuracy"] = 1 - X1["accuracy"]
        if "exact_match" in metrics:
            X1["exact_match"] = 1 - X1["exact_match"]

        X_np = X1.to_numpy()
        best_results = topsis(X_np)
        top_K_results = best_results[:1]
        return X0.iloc[top_K_results]

    def custom_bce_ignore_pads(x, y, pad):

        # https://discuss.pytorch.org/t/custom-loss-functions/29387/95

        t_loss = []
        for j in range(x.shape[0]):
            pad0 = pad[j].int()

            if pad0 > 0:
                x = outputs[1][j][pad0:]
                y = labels_mid[j][pad0:]

            max_val = (-x).clamp_min_(0)
            loss = (
                (1 - y) * x
                + max_val
                + torch.log(torch.exp(-max_val) + torch.exp(-x - max_val))
            )
            t_loss.append(loss.mean())

        return torch.stack(t_loss).mean()

    # Data Generators
    train_dg = DatasetIterator_VICTOR(
        train_df,
        data_path=data_path,
        limit_items=limit_items,
        use_features=use_features,
        use_misfits=use_misfits,
        fine_tune_vf=fine_tune_vf,
        vf_df=vf_df,
        tf_df=tf_df,
        vf_shape=vf_shape,
        tf_shape=vf_shape
        
    )

    valid_dg = DatasetIterator_VICTOR(
        valid_df,
        data_path=data_path,
        limit_items=limit_items,
        use_features=use_features,
        use_misfits=use_misfits,
        fine_tune_vf=fine_tune_vf,
        vf_df=vf_df,
        tf_df=tf_df,
        vf_shape=vf_shape,
        tf_shape=vf_shape
        
    )

    test_dg = DatasetIterator_VICTOR(
        test_df,
        data_path=data_path,
        limit_items=limit_items,
        use_features=use_features,
        use_misfits=use_misfits,
        fine_tune_vf=fine_tune_vf,
        vf_df=vf_df,
        tf_df=tf_df,
        vf_shape=vf_shape,
        tf_shape=vf_shape
        
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

    if use_misfits:
        og_test_dg = DatasetIterator_VICTOR(
            original_test_df,
            data_path=data_path,
            use_misfits=use_misfits,
            limit_items=limit_items,
            fine_tune_vf=fine_tune_vf,
            vf_df=vf_df,
            tf_df=tf_df,
            vf_shape=vf_shape,
            tf_shape=vf_shape
        )

        og_test_dataloader = DataLoader(
            og_test_dg,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    count_experiment = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Training

    for use_MID in use_MID_choices:

        for use_OCr in use_OCr_choices:

            if use_misfits and (not use_OCr and not use_MID):
                print("SKIP EXPERIMENT")
                continue

            w_loss_choices_run = w_loss_choices

            if use_misfits and (not use_OCr or not use_MID):
                w_loss_choices_run = [1]

            for dropout in dropout_choices:
                for tf_layers in tf_layers_choices:
                    for tf_head in tf_head_choices:
                        for tf_dim in tf_dim_choices:
                            for weight_loss in w_loss_choices_run:

                                if use_features == ["images", "texts"]:
                                    emb_size = mm_shape

                                    if fine_tune_vf:
                                        raise Exception(
                                            "Cannot fine tune the image encoder while using texts. Not implemented!"
                                        )

                                elif "images" in use_features and fine_tune_vf:
                                    emb_size = cv_models_params[choose_image_model][
                                        "emb_size"
                                    ]

                                elif "images" in use_features:
                                    emb_size = vf_shape

                                else:
                                    emb_size = tf_shape

                                count_experiment += 1

                                print(
                                    "EXPERIMENT #:",
                                    count_experiment,
                                    "use CPr",
                                    use_OCr,
                                    "use MID",
                                    use_MID,
                                    "tf layers:",
                                    tf_layers,
                                    "tf_dim:",
                                    tf_dim,
                                    "weight_loss",
                                    weight_loss,
                                    "CV_model",
                                    choose_image_model,
                                )

                                # Define model

                                torch.cuda.empty_cache()

                                model = VICTOR(
                                    emb_dim=emb_size,
                                    tf_layers=tf_layers,
                                    tf_head=tf_head,
                                    tf_dim=tf_dim,
                                    activation="gelu",
                                    dropout=dropout,
                                    limit_items=limit_items,
                                    use_OCr=use_OCr,
                                    use_MID=use_MID,
                                    use_misfits=use_misfits,
                                    fine_tune_vf=fine_tune_vf,
                                    choose_cv_model=choose_image_model,
                                    use_cls_token=use_cls_token,
                                    use_extra_attention=use_extra_attention,
                                    device=device
                                )

                                model.to(device)

                                if use_misfits:
                                    criterion_mid = custom_bce_ignore_pads
                                    criterion_regression = nn.MSELoss()

                                else:
                                    criterion = nn.BCEWithLogitsLoss()

                                optimizer = optim.Adam(
                                    model.parameters(), lr=LEARNING_RATE
                                )

                                scheduler = optim.lr_scheduler.StepLR(
                                    optimizer, step_size=10, gamma=0.1, verbose=True
                                )

                                batches_per_epoch = train_dg.__len__() // batch_size

                                PATH = (
                                    "checkpoints_pt/model"
                                    + choose_image_model
                                    + "_"
                                    + str(tf_layers)
                                    + "_"
                                    + str(tf_head)
                                    + "_"
                                    + str(LEARNING_RATE)
                                    + "_"
                                    + str(limit_items)
                                    + ".pt"
                                    if not use_misfits
                                    else "checkpoints_pt/misfit_"
                                    + choose_image_model
                                    + "_"
                                    + ("use_OCr" if use_OCr else "")
                                    + "_"
                                    + ("use_MID" if use_MID else "")
                                    + "_"
                                    + str(tf_layers)
                                    + "_"
                                    + str(tf_head)
                                    + "_"
                                    + str(LEARNING_RATE)
                                    + "_"
                                    + str(limit_items)
                                    + polyvore_version
                                    + "_gen"
                                    + str(generate_per_outfit)
                                    + "_"
                                    + str(weight_loss)
                                    + "_"
                                    + pretraining_method
                                    + ".pt"
                                )

                                history = []
                                has_not_improved_for = 0

                                for epoch in range(EPOCHS):

                                    epoch_start_time = time.time()

                                    running_acc = 0.0
                                    running_loss = 0.0
                                    running_mae = 0.0
                                    running_loss_mid = 0.0
                                    running_loss_reg = 0.0
                                    acc = 0
                                    mae = 0

                                    model.train()

                                    for i, data in enumerate(train_dataloader, 0):

                                        if use_misfits:
                                            (
                                                inputs,
                                                labels_regression,
                                                labels_mid,
                                                pad_positions,
                                            ) = (
                                                data[0].to(device, non_blocking=True),
                                                data[1][0].to(
                                                    device, non_blocking=True
                                                ),
                                                data[1][1].to(
                                                    device, non_blocking=True
                                                ),
                                                data[2].to(device, non_blocking=True),
                                            )
                                        else:
                                            inputs, labels, pad_positions = (
                                                data[0].to(device, non_blocking=True),
                                                data[1].to(device, non_blocking=True),
                                                data[2].to(device, non_blocking=True),
                                            )

                                        optimizer.zero_grad()
                                        outputs = model(inputs, pad_positions)

                                        if use_misfits:

                                            if use_OCr:
                                                loss_reg = criterion_regression(
                                                    outputs[0],
                                                    labels_regression.unsqueeze(1),
                                                )
                                                reg_outputs = (
                                                    torch.clip(outputs[0], 0, 1)
                                                    .cpu()
                                                    .detach()
                                                    .numpy()
                                                )
                                                mae = metrics.mean_absolute_error(
                                                    labels_regression.unsqueeze(1)
                                                    .cpu()
                                                    .detach()
                                                    .numpy(),
                                                    reg_outputs,
                                                )

                                            if use_MID:
                                                loss_mid = criterion_mid(
                                                    outputs[1],
                                                    labels_mid,
                                                    pad_positions,
                                                )
                                                mid_outputs = torch.sigmoid(outputs[1])
                                                mid_outputs = np.round(
                                                    mid_outputs.cpu().detach().numpy()
                                                )
                                                acc = 100 * (
                                                    1
                                                    - round(
                                                        metrics.hamming_loss(
                                                            labels_mid.cpu().numpy(),
                                                            mid_outputs,
                                                        ),
                                                        4,
                                                    )
                                                )

                                            if use_OCr and use_MID:

                                                loss = (
                                                    loss_reg.float()
                                                    + loss_mid.float() * weight_loss
                                                )

                                                running_loss_mid += loss_mid
                                                running_loss_reg += loss_reg

                                            elif use_OCr:
                                                loss = loss_reg.float()
                                                running_loss_reg += loss_reg

                                            else:
                                                loss = loss_mid.float()
                                                running_loss_mid += loss_mid

                                        else:
                                            loss = criterion(
                                                outputs, labels.unsqueeze(1)
                                            )
                                            acc = binary_acc(
                                                outputs, labels.unsqueeze(1)
                                            )

                                        loss.backward()
                                        optimizer.step()

                                        running_loss += loss.item()
                                        running_acc += acc
                                        running_mae += mae

                                        print(
                                            f"[Epoch:{epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f} MSE {running_loss_reg / (i+1):.3f} BCE {running_loss_mid / (i+1):.3f} Acc: {running_acc / (i+1):.3f} Mae: {running_mae/ (i+1):.3f}",
                                            end="\r",
                                        )

                                    print("\nEvaluation:", end=" -> ")

                                    if use_misfits:
                                        cp_result = evaluate_CP_MID(valid_dataloader)

                                        history.append(cp_result)

                                        if use_OCr and use_MID:
                                            metrics_list = [
                                                "MAE",
                                                "exact_match",
                                            ]  # accuracy
                                        elif use_OCr:
                                            metrics_list = ["MAE"]
                                        elif use_MID:
                                            metrics_list = ["exact_match"]  # accuracy

                                        best_index = choose_best_model(
                                            pd.DataFrame(history), metrics=metrics_list
                                        ).index[0]

                                        if epoch == best_index:
                                            print("Checkpoint!\n")
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

                                    else:
                                        cp_result = evaluate_cp(valid_dataloader)
                                        if epoch == 0 or cp_result["AUC"] > max(
                                            [x["CP_AUC"] for x in history]
                                        ):
                                            print("Checkpoint!\n")
                                            torch.save(
                                                {
                                                    "epoch": epoch,
                                                    "model_state_dict": model.state_dict(),
                                                    "optimizer_state_dict": optimizer.state_dict(),
                                                },
                                                PATH,
                                            )

                                            result = {}
                                            result["mode"] = "validation"
                                            result["CP_AUC"] = cp_result["AUC"]

                                            has_not_improved_for = 0
                                        else:
                                            has_not_improved_for += 1

                                        history.append(result)

                                    print("\n")
                                    scheduler.step()

                                    if has_not_improved_for >= early_stop_epochs:
                                        break

                                print(
                                    "Finished Training. Loading the best model from checkpoints."
                                )

                                checkpoint = torch.load(PATH)
                                model.load_state_dict(checkpoint["model_state_dict"])
                                optimizer.load_state_dict(
                                    checkpoint["optimizer_state_dict"]
                                )
                                epoch = checkpoint["epoch"]

                                if use_misfits:
                                    print("Evaluate CPr and MID")
                                    MID_result = evaluate_CP_MID(test_dataloader)

                                    print("Evalute CPb")
                                    cp_result = evaluate_cp(og_test_dataloader)

                                else:
                                    print("Evaluation on testing data")
                                    cp_result = evaluate_cp(test_dataloader)

                                result = {}
                                result["mode"] = "testing_misfit"
                                result["choose_image_model"] = choose_image_model
                                result["polyvore_version"] = polyvore_version
                                result["limit_items"] = limit_items
                                result["use_MID"] = use_MID
                                result["use_OCr"] = use_OCr
                                result["generate_items"] = generate_per_outfit
                                result["CP_AUC"] = cp_result["AUC"]

                                result["dropout"] = dropout
                                result["tf_layers"] = tf_layers
                                result["tf_head"] = tf_head
                                result["tf_dim"] = tf_dim

                                result["MAE"] = MID_result["MAE"] if use_misfits else 1
                                result["MSE"] = MID_result["MSE"] if use_misfits else 1
                                result["exact_match"] = (
                                    MID_result["exact_match"] if use_misfits else 0
                                )
                                result["accuracy"] = (
                                    MID_result["accuracy"] if use_misfits else 0
                                )

                                print(result)
                                result["history"] = history
                                result["use_features"] = use_features
                                result["fine_tune_vf"] = fine_tune_vf
                                result["cls_token"] = use_cls_token
                                result["learning_rate"] = LEARNING_RATE
                                result["batch_size"] = batch_size
                                result["checkpoint_path"] = PATH
                                result["use_extra_attention"] = use_extra_attention
                                result["weight_loss"] = weight_loss
                                result["NOTES"] = notes
                                result["FLIP_emb_size"] = FLIP_emb_size
                                result["FLIP_learning_rate"] = FLIP_learning_rate

                                if save_results:
                                    if not os.path.isdir(data_path + "results"):
                                        os.mkdir(data_path + "results")

                                    save_results_csv(
                                        "data_benchmark/polyvore/polyvore_outfits/results/",
                                        save_results_to,
                                        result,
                                    )
                                del model
                                gc.collect()
                                torch.cuda.empty_cache()

                                print("sleep for 1 minute")
                                time.sleep(60)