#!/bin/bash
echo "📦 Initializing the FPK package"

CONTEXT_PYTHON_SCRIPT="/tmp/context_python_script.py"
VERIFY_MODE=${VERIFY_MODE:-false}
VENV_PYTHON="python3"
EXEC_SCRIPT="/tmp/exec_python_script.py"

rm -f "$CONTEXT_PYTHON_SCRIPT" "$EXEC_SCRIPT"
touch "$CONTEXT_PYTHON_SCRIPT" "$EXEC_SCRIPT"

trap 'rm -f "$CONTEXT_PYTHON_SCRIPT" "$EXEC_SCRIPT"' EXIT

log_error() {
  echo "❌ $1"
  exit 1
}

strip_html_tags() {
  echo "$1" | sed 's/<[^>]*>//g'
}

decode_html_entities() {
  echo "$1" | sed "s/&#39;/'/g" | sed "s/&quot;/\"/g" | sed "s/&gt;/>/g" | sed "s/&lt;/</g" | sed "s/&amp;/&/g"
}

part_python() {
  local code="$1"
  code=$(strip_html_tags "$code")
  code=$(decode_html_entities "$code")

  if [[ -z "$code" ]]; then
    log_error "Python code block is empty."
  fi

  if [[ "$VERIFY_MODE" != "true" ]]; then
    context_code=$(echo "$code" | grep -vE 'print\(|subprocess\.run')
    execution_code=$(echo "$code" | grep -E 'print\(|subprocess\.run')

    echo "$context_code" >>"$CONTEXT_PYTHON_SCRIPT"

    echo "try:" >"$EXEC_SCRIPT"
    sed 's/^/    /' "$CONTEXT_PYTHON_SCRIPT" >>"$EXEC_SCRIPT"
    echo "$execution_code" | sed 's/^/    /' >>"$EXEC_SCRIPT"
    echo "except Exception as e:" >>"$EXEC_SCRIPT"
    echo "    print(e)" >>"$EXEC_SCRIPT"

    if ! output=$("$VENV_PYTHON" "$EXEC_SCRIPT" 2>&1); then
      log_error "Error executing Python script: $output"
    else
      echo "$output"
    fi
  else
    echo "Verifying Python code..."
    if "$VENV_PYTHON" -m py_compile "$EXEC_SCRIPT"; then
      echo "✅ OK"
    else
      log_error "❌ Fail"
    fi
  fi
}

part_bash() {
  local code="$1"
  code=$(strip_html_tags "$code")

  if [[ -z "$code" ]]; then
    log_error "Bash code block is empty."
  fi

  if [[ "$VERIFY_MODE" != "true" ]]; then
    code=$(echo "$code" | sed 's/\\\$/\$/g')

    if ! source /dev/stdin <<<"$code"; then
      log_error "Error executing Bash script."
    fi
  else
    echo "Verifying Bash code..."
    if bash -n <<<"$code"; then
      echo "✅ OK"
    else
      log_error "❌ Fail"
    fi
  fi
}

if [[ "$VERIFY_MODE" == "true" ]]; then
  echo "🔍 Verification mode enabled"
fi
