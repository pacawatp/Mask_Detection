import torch
from PIL import Image

# Change model name and weight
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights


def to_onnx():
    # Step 1: Initialize model with the best available weights
    weights = RegNet_X_400MF_Weights.DEFAULT
    torch_model = regnet_x_400mf(weights=weights)
    torch_model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    dog = Image.open("./dog.jpg").convert("RGB")

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(dog).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = torch_model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(weights.meta["categories"])
    print(f"{category_name}: {100 * score:.1f}%")

    torch_out = torch_model(batch)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        batch,  # model input (or a tuple for multiple inputs)
        "model_name.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization.
        # example i = 320 * 200 * 32 convert to i = 2048000
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )


if __name__ == "__main__":
    to_onnx()
