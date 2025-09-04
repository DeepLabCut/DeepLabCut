from deeplabcut.modelzoo import build_weight_init

superanimal_name = "superanimal_humanbody"
model_name = "rtmpose_x"
detector_name = 'fasterrcnn_resnet50_fpn'

weight_init = build_weight_init(
    cfg="/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/config.yaml",
    super_animal=superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    with_decoder=True,
)

print("Yes!")