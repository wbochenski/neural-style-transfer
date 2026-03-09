import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
layer_map = {
    1: "conv1_1",
    3: "conv1_2",
    6: "conv2_1",
    8: "conv2_2",
    11: "conv3_1",
    13: "conv3_2",
    15: "conv3_3",
    17: "conv3_4",
    20: "conv4_1",
    22: "conv4_2",
    24: "conv4_3",
    26: "conv4_4",
    29: "conv5_1",
    31: "conv5_2",
    33: "conv5_3",
    35: "conv5_4",
}