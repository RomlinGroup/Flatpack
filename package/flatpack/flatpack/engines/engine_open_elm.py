from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenELMEngine:
    def __init__(self, model_name, n_ctx=4096, verbose=False, hf_access_token=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_access_token,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_access_token,
            trust_remote_code=True
        )

        self.n_ctx = n_ctx
        self.verbose = verbose

    def generate_response(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\n"
        inputs = self.tokenizer(prompt, return_tensors='pt')

        output = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.n_ctx,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
