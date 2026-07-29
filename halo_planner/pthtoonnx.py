import torch 
import torch.nn as nn 
import torch.nn.functional as F


from testcnn import BasicCNN

def main():
    model_path = "./BasicCNNonMNIST.pth"
    onnx_path = "./BasicCNNonMNIST.onnx"


    print("Loading the pytorch weights first....")

    model = torch.load(model_path, weights_only=True)

    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    print(f"Extracting to {onnx_path}....")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch_size'},
            'output' : {0, 'batch_size'}
        }
    )

    print("Onnx Exported") 

if __name__ == "__main__":
    main()

