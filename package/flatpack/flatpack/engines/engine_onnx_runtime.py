import onnxruntime as ort
import numpy as np


class ONNXRuntimeEngine:
    def __init__(self, model_path, providers=['CPUExecutionProvider'], n_threads=6):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.session.set_providers(providers, [{'num_threads': n_threads}])

    def generate_response(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\n"

        # Assuming the model takes a single string input and outputs a string
        inputs = {self.session.get_inputs()[0].name: np.array([prompt], dtype=np.str_)}

        # Run the model
        outputs = self.session.run(None, inputs)

        # Assuming the output is the first output node
        response = outputs[0][0]
        return response
