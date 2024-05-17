from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenELMEngine:
    def __init__(self, model_name, n_ctx=1024, verbose=False, hf_access_token=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_access_token,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            use_auth_token=hf_access_token,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.n_ctx = n_ctx
        self.verbose = verbose

    def generate_response(self, prompt, generate_kwargs=None):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=self.n_ctx)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")

        if generate_kwargs is None:
            generate_kwargs = {}

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.n_ctx,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
