from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenELMEngine:
    def __init__(self, model_name, n_ctx=4096, n_threads=6, verbose=False):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.n_ctx = n_ctx
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

    def generate_response(self, context, question):
        logging.debug(f"Generating response for context: {context} and question: {question}")
        prompt = f"Context: {context}\nQuestion: {question}\n"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.n_ctx,
            repetition_penalty=1.0
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        logging.debug(f"Generated response: {response}")
        return response
