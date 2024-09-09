#!/bin/bash
set -euo pipefail
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON=${VENV_PYTHON:-python}
CONTEXT_PYTHON_SCRIPT="/var/folders/s6/g1h7_qhn72xfxpyf5wrqngpc0000gn/T/tmp07tzj5rp"
EVAL_BUILD="$(dirname "$SCRIPT_DIR")/test/eval_build.json"
EXEC_PYTHON_SCRIPT="/var/folders/s6/g1h7_qhn72xfxpyf5wrqngpc0000gn/T/tmp7h2p7w96"
CURR=0
last_count=5
trap 'rm -f "$CONTEXT_PYTHON_SCRIPT" "$EXEC_PYTHON_SCRIPT"; exit' EXIT INT TERM
rm -f "$CONTEXT_PYTHON_SCRIPT" "$EXEC_PYTHON_SCRIPT"
touch "$CONTEXT_PYTHON_SCRIPT" "$EXEC_PYTHON_SCRIPT"
datetime=$(date -u +"%Y-%m-%d %H:%M:%S")
DATA_FILE="$(dirname "$SCRIPT_DIR")/test/eval_data.json"
echo '[]' > "$DATA_FILE"

function log_data() {
    local part_number="$1"
    local new_files=$(find "$SCRIPT_DIR" -type f -newer "$DATA_FILE" \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.txt' \) ! -path '*/bin/*' ! -path '*/lib/*')
    if [ -n "$new_files" ]; then
        local log_entries="[]"
        local temp_file=$(mktemp)
        for file in $new_files; do
            local mime_type=$(file --mime-type -b "$file")
            local web=$(basename "$file")
            local json_entry="{\"eval\": $part_number, \"file\": \"$file\", \"public\": \"/output/$web\", \"type\": \"$mime_type\"}"
            log_entries=$(echo "$log_entries" | jq ". + [$json_entry]")
        done
        jq ". + $log_entries" "$DATA_FILE" > "$temp_file" && mv "$temp_file" "$DATA_FILE"
    fi
    touch "$DATA_FILE"
}

function update_eval_build() {
    local curr="$1"
    local eval="$2"
    echo "{
        \"curr\": $curr,
        \"last\": $last_count,
        \"eval\": $eval,
        \"datetime\": \"$datetime\"
    }" > "$EVAL_BUILD"
}

update_eval_build "$CURR" 1

BASH_VAR="Hello from Bash"
echo "Setting BASH_VAR: $BASH_VAR"
((CURR++))
log_data "$CURR"
if [ "$CURR" -eq "$last_count" ]; then
    EVAL="null"
else
    EVAL=$((CURR + 1))
fi
update_eval_build "$CURR" "$EVAL"

echo "python_var = \"Hello from Python\"" >> "$CONTEXT_PYTHON_SCRIPT"
echo "try:" > "$EXEC_PYTHON_SCRIPT"
sed 's/^/    /' "$CONTEXT_PYTHON_SCRIPT" >> "$EXEC_PYTHON_SCRIPT"
echo "except Exception as e:" >> "$EXEC_PYTHON_SCRIPT"
echo "    print(e)" >> "$EXEC_PYTHON_SCRIPT"
echo "    import sys; sys.exit(1)" >> "$EXEC_PYTHON_SCRIPT"
$VENV_PYTHON "$EXEC_PYTHON_SCRIPT"
((CURR++))
log_data "$CURR"
if [ "$CURR" -eq "$last_count" ]; then
    EVAL="null"
else
    EVAL=$((CURR + 1))
fi
update_eval_build "$CURR" "$EVAL"

echo "Accessing BASH_VAR from previous cell: $BASH_VAR"
BASH_VAR="Updated in second bash cell"
echo "Updated BASH_VAR: $BASH_VAR"
((CURR++))
log_data "$CURR"
if [ "$CURR" -eq "$last_count" ]; then
    EVAL="null"
else
    EVAL=$((CURR + 1))
fi
update_eval_build "$CURR" "$EVAL"

echo "try:" > "$EXEC_PYTHON_SCRIPT"
sed 's/^/    /' "$CONTEXT_PYTHON_SCRIPT" >> "$EXEC_PYTHON_SCRIPT"
echo "print(python_var)" | sed 's/^/    /' >> "$EXEC_PYTHON_SCRIPT"
echo "except Exception as e:" >> "$EXEC_PYTHON_SCRIPT"
echo "    print(e)" >> "$EXEC_PYTHON_SCRIPT"
echo "    import sys; sys.exit(1)" >> "$EXEC_PYTHON_SCRIPT"
$VENV_PYTHON "$EXEC_PYTHON_SCRIPT"
((CURR++))
log_data "$CURR"
if [ "$CURR" -eq "$last_count" ]; then
    EVAL="null"
else
    EVAL=$((CURR + 1))
fi
update_eval_build "$CURR" "$EVAL"

echo "from transformers import AutoTokenizer, AutoModelForCausalLM

device = \"mps\"
model_path = \"01-ai/Yi-Coder-9B-Chat\"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=\"auto\").eval()

prompt = \"Write a quick sort algorithm.\"
messages = [
    {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
    {\"role\": \"user\", \"content\": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors=\"pt\").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]" >> "$CONTEXT_PYTHON_SCRIPT"
echo "try:" > "$EXEC_PYTHON_SCRIPT"
sed 's/^/    /' "$CONTEXT_PYTHON_SCRIPT" >> "$EXEC_PYTHON_SCRIPT"
echo "print(response)" | sed 's/^/    /' >> "$EXEC_PYTHON_SCRIPT"
echo "except Exception as e:" >> "$EXEC_PYTHON_SCRIPT"
echo "    print(e)" >> "$EXEC_PYTHON_SCRIPT"
echo "    import sys; sys.exit(1)" >> "$EXEC_PYTHON_SCRIPT"
$VENV_PYTHON "$EXEC_PYTHON_SCRIPT"
((CURR++))
log_data "$CURR"
if [ "$CURR" -eq "$last_count" ]; then
    EVAL="null"
else
    EVAL=$((CURR + 1))
fi
update_eval_build "$CURR" "$EVAL"

