# convert_model_to_onnx

```bash
./convert_model_to_onnx.sh gpt2
```

```bash
./upload_to_hf.sh gpt2_onnx gpt2
```

```bash
python append_onnx.py --folder_path "transformers.js/models/your-model" --repo_id "your-repo"
```