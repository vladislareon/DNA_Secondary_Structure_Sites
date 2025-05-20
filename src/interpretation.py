import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torch
import captum
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTR_METHODS_INV = {
    "IntegratedGradients": IntegratedGradients,
    "GradientShap": GradientShap,
    "DeepLift": DeepLift,
    "Saliency": Saliency,
    "InputXGradient": InputXGradient,
    "GuidedBackpropagation": GuidedBackprop,
    "Deconvolution": Deconvolution,
}


def get_mean_interpretation(data_loader, mean1, cnt, model, width, algorithm):
    explain = ATTR_METHODS_INV[algorithm](model)
    for x, y_true in tqdm(data_loader):
        x, y_true = x.to(device), y_true.to(device).long()
        output = model(x)
        pred = torch.argmax(output, dim=1).reshape(1, width)
        idxs = []
        for i in range(width):
            if pred[0][i] == y_true[0][i] and y_true[0][i] == 1:
                idxs.append(i)

        if algorithm == "IntegratedGradients":
            explanation = explain.attribute(x, target=1, n_steps=1)
        elif algorithm == "DeepLift":
            explanation = torch.zeros_like(x)
            cnt_expl = 0
            for index in range(x.shape[1]):
                if y_true[0][index] == 1:
                    attribution = explain.attribute(x, target=(index, 1))
                    explanation += attribution
                    cnt_expl += 1
            explanation /= cnt_expl

        elif algorithm == "GradientShap":
            explanation = torch.zeros_like(x)
            cnt_expl = 0
            for index in range(x.shape[1]):
                if y_true[0][index] == 1:
                    attribution = explain.attribute(
                        x, target=(index, 1), baselines=torch.zeros_like(x)
                    )
                    explanation += attribution
                    cnt_expl += 1
            explanation /= cnt_expl

        else:
            explanation = explain.attribute(x, target=1)
        explanation = torch.squeeze(explanation, dim=0)

        if explanation[idxs, :].shape != (0, 1950):
            explanation = torch.mean(explanation[idxs, :], dim=0)
            explanation = explanation.cpu().detach().numpy()
            mean1 += explanation
            cnt += 1
    return mean1, cnt


def cnn_interpretation_pipeline(
    model,
    loader_test,
    loader_train,
    width,
    save_filename,
    algorithm="IntegratedGradients",
    need_return=1,
):
    mean1 = np.zeros(1950, dtype=float)
    cnt = 0
    mean1, cnt = get_mean_interpretation(
        loader_test, mean1, cnt, model, width, algorithm
    )

    mean = mean1 / cnt
    mean = torch.from_numpy(mean)
    print(f"Averaged tensor shape: {mean.shape}")
    print(f"Averaged tensor: {mean}")

    torch.save(mean, f"{save_filename}.pt")
    print("Interpretation result is an averaged tensor. It is saved as:")
    print(f"{save_filename}.pt")

    if need_return:
        return mean
    else:
        return
