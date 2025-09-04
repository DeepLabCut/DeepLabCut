from deeplabcut.modelzoo import build_weight_init

superanimal_name = "superanimal_humanbody"
model_name = "rtmpose_x"
detector_name = 'fasterrcnn_resnet50_fpn_v2'

config = "/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/config.yaml"

weight_init = build_weight_init(
    cfg=config,
    super_animal=superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    with_decoder=False,
)

import deeplabcut

deeplabcut.create_training_dataset_from_existing_split(
    config=config,
    from_shuffle=0,
    shuffles=[15],
    engine=deeplabcut.Engine.PYTORCH,
    net_type=model_name,
    detector_type=None,
    weight_init=weight_init,
    userfeedback=False,
)
# Here, I should think: what model config do we want, especially regarding the detector?


deeplabcut.train_network(
    config=config,
    detector_epochs=0,
    epochs=50,
    save_epochs=10,
    batch_size=64,  # if you get a CUDA OOM error when training on a GPU, reduce to 32, 16, ...!
    displayiters=10,
    shuffle=superanimal_transfer_learning_shuffle,
)