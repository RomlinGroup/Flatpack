#!/bin/bash
set -euo pipefail

echo "📦 Initializing the FPK package"

CONTEXT_PYTHON_SCRIPT="/tmp/context_python_script.py"
EVAL_BUILD="$SCRIPT_DIR/eval_build.json"
EXEC_PYTHON_SCRIPT="/tmp/exec_python_script.py"

VENV_PYTHON="python3"
VERIFY_MODE=${VERIFY_MODE:-false}

CURR=0
LAST=0

cleanup() {
  rm -f "$CONTEXT_PYTHON_SCRIPT" "$EXEC_PYTHON_SCRIPT"
}

trap cleanup EXIT

rm -f "$CONTEXT_PYTHON_SCRIPT" "$EXEC_PYTHON_SCRIPT"
touch "$CONTEXT_PYTHON_SCRIPT" "$EXEC_PYTHON_SCRIPT"

CUSTOM_SH="$SCRIPT_DIR/custom.sh"

if [[ -f "$CUSTOM_SH" && -r "$CUSTOM_SH" ]]; then
  LAST=$(grep -Ec 'part_python|part_bash' "$CUSTOM_SH" || true)
  LAST=${LAST:-0}
else
  echo "File custom.sh does not exist or is not readable"
fi

strip_html_tags() {
  echo "$1" | sed 's/<[^>]*>//g'
}

decode_html_entities() {
  echo "$1" | sed "s/&#39;/'/g" | sed "s/&quot;/\"/g" | sed "s/&gt;/>/g" | sed "s/&lt;/</g" | sed "s/&amp;/&/g"
}

eval_build() {
  local datetime
  datetime=$(date -u +"%Y-%m-%d %H:%M:%S")
  local eval

  if [ "$CURR" -eq "$LAST" ]; then
    eval="null"
  else
    eval=$((CURR + 1))
  fi

  echo "{\"curr\": $CURR, \"last\": $LAST, \"eval\": $eval, \"datetime\": \"$datetime\"}" >"$EVAL_BUILD"
}

part_python() {
  local code="$1"
  code=$(strip_html_tags "$code")
  code=$(decode_html_entities "$code")

  if [[ -z "$code" ]]; then
    echo "Python code block is empty."
    exit 1
  fi

  if [[ "$VERIFY_MODE" != "true" ]]; then
    context_code=$(echo "$code" | grep -vE 'print\(|subprocess\.run')
    execution_code=$(echo "$code" | grep -E 'print\(|subprocess\.run')

    echo "$context_code" >>"$CONTEXT_PYTHON_SCRIPT"

    echo "try:" >"$EXEC_PYTHON_SCRIPT"
    sed 's/^/    /' "$CONTEXT_PYTHON_SCRIPT" >>"$EXEC_PYTHON_SCRIPT"
    echo "$execution_code" | sed 's/^/    /' >>"$EXEC_PYTHON_SCRIPT"
    echo "except Exception as e:" >>"$EXEC_PYTHON_SCRIPT"
    echo "    print(e)" >>"$EXEC_PYTHON_SCRIPT"
    echo "    import sys; sys.exit(1)" >>"$EXEC_PYTHON_SCRIPT"

    if output=$("$VENV_PYTHON" "$EXEC_PYTHON_SCRIPT" 2>&1); then
      echo "$output"
      ((CURR++))
    else
      exit 1
    fi

    eval_build

  else
    echo "Verifying Python code..."
  fi
}

part_bash() {
  local code="$1"
  code=$(strip_html_tags "$code")

  if [[ -z "$code" ]]; then
    echo "Bash code block is empty."
    exit 1
  fi

  if [[ "$VERIFY_MODE" != "true" ]]; then
    code=$(echo "$code" | sed 's/\\\$/\$/g')

    if source /dev/stdin <<<"$code" 2>/dev/null; then
      ((CURR++))
    else
      exit 0
    fi

    eval_build

  else
    echo "Verifying Bash code..."
  fi
}

if [[ "$VERIFY_MODE" == "true" ]]; then
  echo "🔍 Verification mode enabled"
fi
