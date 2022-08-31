from FLIP import train_flip
from VICTOR import run_experiment

# Where the polyvore dataset is stored. This must be defined by the user.
data_path = "data_benchmark/polyvore/polyvore_outfits/"

# First we train FLIP with four different computer vision models
for choose_image_model in [
    "resnet18",
    "tf_efficientnetv2_b3",
    "vit_base_patch32_224",
    "mixer_b16_224",
]:
    train_flip(choose_image_model=choose_image_model, data_path=data_path)

# Ablation study for VICTOR[MTL], VICTOR[OCr] and VICTOR[MID]
# OCr stands for Outfit Compatibility as regression
# MID stands for Mismatching Item Detection
# MTL stands for multi-task-learning which combines both OCr and MID

for m in [2, 4]:
    for polyvore_version in ["disjoint", "nondisjoint"]:
        for pretraining_method in ["FLIP", "ImageNet"]:
            for choose_image_model in [
                "resnet18",
                "mixer_b16_224",
                "vit_base_patch32_224",
                "tf_efficientnetv2_b3",
            ]:

                w_loss_choices = [0.2, 0.5, 1, 2]
                run_experiment(
                    choose_gpu=0,
                    data_path=data_path,
                    use_MID_choices=[True, False],
                    use_OCr_choices=[True, False],
                    w_loss_choices=w_loss_choices,
                    save_results_to="results_MISFITS",
                    use_misfits=True,
                    pretraining_method=pretraining_method,
                    polyvore_version=polyvore_version,
                    choose_image_model=choose_image_model,
                    choose_text_model="CLIP_Transformer",
                    use_features=["images"],
                    generate_per_outfit=m,
                    EPOCHS=20,
                    LEARNING_RATE=1e-4,
                )


# Training VICTOR[OCb]
# OCb stands for Outfit Compatibility as binary classification.
for polyvore_version in ["disjoint", "nondisjoint"]:
    for pretraining_method in ["FLIP", "ImageNet"]:
        for choose_image_model in [
            "resnet18",
            "mixer_b16_224",
            "vit_base_patch32_224",
            "tf_efficientnetv2_b3",
        ]:

            run_experiment(
                choose_gpu=0,
                data_path=data_path,
                use_MID_choices=[False],
                use_OCr_choices=[False],
                w_loss_choices=[1],
                save_results_to="results_MISFITS",
                use_misfits=False,
                pretraining_method=pretraining_method,
                polyvore_version=polyvore_version,
                choose_image_model=choose_image_model,
                choose_text_model="CLIP_Transformer",
                use_features=["images"],
                generate_per_outfit=0,
                EPOCHS=20,
                LEARNING_RATE=1e-4,
            )

# Training VICTOR[MTL] with multi-modal features (image+text)
for m in [2, 4]:
    for polyvore_version in ["disjoint", "nondisjoint"]:
        for pretraining_method in ["FLIP"]:
            for choose_image_model in [
                "resnet18",
                "mixer_b16_224",
                "vit_base_patch32_224",
                "tf_efficientnetv2_b3",
            ]:

                w_loss_choices = [0.2, 1]
                run_experiment(
                    choose_gpu=0,
                    data_path=data_path,
                    use_MID_choices=[True],
                    use_OCr_choices=[True],
                    w_loss_choices=w_loss_choices,
                    save_results_to="results_MISFITS",
                    use_misfits=True,
                    pretraining_method=pretraining_method,
                    polyvore_version=polyvore_version,
                    choose_image_model=choose_image_model,
                    choose_text_model="CLIP_Transformer",
                    use_features=["images", "texts"],
                    generate_per_outfit=m,
                    EPOCHS=20,
                    LEARNING_RATE=1e-4,
                )

# Training VICTOR[MTL] with text only inputs
for m in [2, 4]:
    for polyvore_version in ["nondisjoint"]:
        for pretraining_method in ["FLIP"]:
            for choose_image_model in ["resnet18"]:
                # ResNet18 features are not being used here but the code requires the name of a model to load the features.
                # FLIP's text encoder is not fine-tuned during pre-training.
                # Therefore there is no difference between using text features from FLIP trained with resnet18 or any other CV model.

                w_loss_choices = [0.2, 1]
                run_experiment(
                    choose_gpu=0,
                    data_path=data_path,
                    use_MID_choices=[True],
                    use_OCr_choices=[True],
                    w_loss_choices=w_loss_choices,
                    save_results_to="results_MISFITS",
                    use_misfits=True,
                    pretraining_method=pretraining_method,
                    polyvore_version=polyvore_version,
                    choose_image_model=choose_image_model,
                    choose_text_model="CLIP_Transformer",
                    use_features=["texts"],
                    generate_per_outfit=m,
                    EPOCHS=20,
                    LEARNING_RATE=1e-4,
                )