import onnx
from onnxsim import simplify
import os


def optimize_onnx_file(onnxfile):
    model = onnx.load(onnxfile)
    model_simp, check = simplify(model)

    if check:
        onnx.save(model_simp, onnxfile)
    else:
        print(f"Warning: Simplification failed for {onnxfile}. Skipping...")


def main(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".onnx"):
            filepath = os.path.join(folder_path, filename)
            optimize_onnx_file(filepath)
            print(f"Optimized {filename}")


if __name__ == "__main__":
    folder_path = input("Enter the folder path containing the ONNX files: ")
    main(folder_path)
