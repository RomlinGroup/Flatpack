import numpy as np
import onnxruntime as ort


class ONNXRuntimeEngine:
    def __init__(self, model_path, providers=['CPUExecutionProvider'], n_threads=6):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.session.set_providers(providers, [{'num_threads': n_threads}])

    def generate_response(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\n"

        inputs = {self.session.get_inputs()[0].name: np.array([prompt], dtype=np.str_)}
        outputs = self.session.run(None, inputs)

        response = outputs[0][0]
        return response
