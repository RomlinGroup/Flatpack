#!/bin/bash
echo "📦 Initializing the FPK package"

CONTEXT_PYTHON_SCRIPT="/tmp/context_python_script.py"
VERIFY_MODE=${VERIFY_MODE:-false}
VENV_PYTHON="python3"

rm -f "$CONTEXT_PYTHON_SCRIPT"
touch "$CONTEXT_PYTHON_SCRIPT"

trap 'rm -f "$CONTEXT_PYTHON_SCRIPT"' EXIT

log_error() {
  echo "❌ $1"
  exit 1
}

strip_html_tags() {
  echo "$1" | sed 's/<[^>]*>//g'
}

part_python() {
  local code="$1"
  local exec_script='/tmp/exec_python_script.py'

  code=$(strip_html_tags "$code")
  code=$(echo "$code" | sed "s/&#39;/'/g" | sed "s/&quot;/\"/g" | sed "s/&gt;/>/g" | sed "s/&lt;/</g" | sed "s/&amp;/&/g")

  if [[ -z "$code" ]]; then
    log_error "Python code block is empty."
  fi

  if [[ "$VERIFY_MODE" != "true" ]]; then

    context_code=$(echo "$code" | grep -vE 'print\(|subprocess\.run')
    execution_code=$(echo "$code" | grep -E 'print\(|subprocess\.run')

    echo "$context_code" >>"$CONTEXT_PYTHON_SCRIPT"

    echo "try:" >"$exec_script"
    cat "$CONTEXT_PYTHON_SCRIPT" | sed 's/^/    /' >>"$exec_script"
    echo "$execution_code" | sed 's/^/    /' >>"$exec_script"
    echo "except Exception as e:" >>"$exec_script"
    echo "    print(e)" >>"$exec_script"

  else

    echo "Verifying Python code..."
    if "$VENV_PYTHON" -m py_compile "$exec_script"; then
      echo "✅ OK"
    else
      echo "❌ Fail"
    fi

  fi

  #if ! "$VENV_PYTHON" -m py_compile "$exec_script"; then
  #  log_error "Invalid Python code. Exiting..."
  #fi

  if ! output=$("$VENV_PYTHON" "$exec_script" 2>&1); then
    log_error "Error executing Python script: $output"
  else
    if [[ "$VERIFY_MODE" != "true" ]]; then
      echo "$output"
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
      echo "❌ Fail"
    fi

  fi
}

if [[ "$VERIFY_MODE" == "true" ]]; then
  echo "🔍 Verification mode enabled"
fi
