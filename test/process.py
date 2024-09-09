import json
import logging
import subprocess
import tempfile
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from pathlib import Path

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def strip_html(content: str) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()


def create_temp_sh(custom_sh_path: Path, temp_sh_path: Path, use_euxo: bool = False):
    try:
        with custom_sh_path.open('r') as infile:
            script = infile.read()

        script = strip_html(script)

        parts = []
        lines = script.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith(
                    ('part_bash """', 'part_python """', 'disabled part_bash """', 'disabled part_python """')):
                start_line = i
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('"""'):
                    i += 1
                end_line = i + 1 if i < len(lines) else i
                parts.append('\n'.join(lines[start_line:end_line]).strip())
            i += 1

        last_count = sum(1 for part in parts if not part.startswith('disabled'))

        temp_sh_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as context_python_script:
            context_python_script_path = Path(context_python_script.name)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as exec_python_script:
            exec_python_script_path = Path(exec_python_script.name)

        with temp_sh_path.open('w') as outfile:
            outfile.write("#!/bin/bash\n")
            outfile.write(f"set -{'eux' if use_euxo else 'eu'}o pipefail\n")

            # Add the SCRIPT_DIR export line
            outfile.write('export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')

            # Handle the $VENV_PYTHON variable (this should be defined or passed in)
            outfile.write("VENV_PYTHON=${VENV_PYTHON:-python}\n")

            outfile.write(f"CONTEXT_PYTHON_SCRIPT=\"{context_python_script_path}\"\n")

            outfile.write("EVAL_BUILD=\"$(dirname \"$SCRIPT_DIR\")/test/eval_build.json\"\n")
            # outfile.write("EVAL_BUILD=\"$(dirname \"$SCRIPT_DIR\")/web/eval_build.json\"\n")

            outfile.write(f"EXEC_PYTHON_SCRIPT=\"{exec_python_script_path}\"\n")
            outfile.write("CURR=0\n")
            outfile.write(f"last_count={last_count}\n")

            outfile.write("trap 'rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"; exit' EXIT INT TERM\n")
            outfile.write("rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")
            outfile.write("touch \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")

            outfile.write("datetime=$(date -u +\"%Y-%m-%d %H:%M:%S\")\n")

            outfile.write("DATA_FILE=\"$(dirname \"$SCRIPT_DIR\")/test/eval_data.json\"\n")
            # outfile.write("DATA_FILE=\"$(dirname \"$SCRIPT_DIR\")/web/eval_data.json\"\n")

            outfile.write("echo '[]' > \"$DATA_FILE\"\n\n")

            outfile.write("function log_data() {\n")
            outfile.write("    local part_number=\"$1\"\n")

            outfile.write(
                "    local new_files=$(find \"$SCRIPT_DIR\" -type f -newer \"$DATA_FILE\" \\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.txt' \\) ! -path '*/bin/*' ! -path '*/lib/*')\n"
            )
            outfile.write("    if [ -n \"$new_files\" ]; then\n")
            outfile.write("        local log_entries=\"[]\"\n")
            outfile.write("        local temp_file=$(mktemp)\n")
            outfile.write("        for file in $new_files; do\n")
            outfile.write("            local mime_type=$(file --mime-type -b \"$file\")\n")
            outfile.write("            local web=$(basename \"$file\")\n")
            outfile.write(
                "            local json_entry=\"{\\\"eval\\\": $part_number, \\\"file\\\": \\\"$file\\\", \\\"public\\\": \\\"/output/$web\\\", \\\"type\\\": \\\"$mime_type\\\"}\"\n"
            )
            outfile.write("            log_entries=$(echo \"$log_entries\" | jq \". + [$json_entry]\")\n")
            outfile.write("        done\n")
            outfile.write(
                "        jq \". + $log_entries\" \"$DATA_FILE\" > \"$temp_file\" && mv \"$temp_file\" \"$DATA_FILE\"\n")
            outfile.write("    fi\n")
            outfile.write("    touch \"$DATA_FILE\"\n")
            outfile.write("}\n\n")

            outfile.write("function update_eval_build() {\n")
            outfile.write("    local curr=\"$1\"\n")
            outfile.write("    local eval=\"$2\"\n")
            outfile.write("    echo \"{\n")
            outfile.write("        \\\"curr\\\": $curr,\n")
            outfile.write("        \\\"last\\\": $last_count,\n")
            outfile.write("        \\\"eval\\\": $eval,\n")
            outfile.write("        \\\"datetime\\\": \\\"$datetime\\\"\n")
            outfile.write("    }\" > \"$EVAL_BUILD\"\n")
            outfile.write("}\n\n")

            outfile.write("update_eval_build \"$CURR\" 1\n\n")

            for part in parts:
                part_lines = part.splitlines()
                if len(part_lines) < 2:
                    continue

                header = part_lines[0].strip()
                code_lines = part_lines[1:-1]

                if header.startswith('disabled'):
                    continue

                language = 'bash' if 'part_bash' in header else 'python' if 'part_python' in header else None
                code = '\n'.join(code_lines).strip().replace('\\"', '"')

                if language == 'bash':
                    code = code.replace('$$', r'$$$$')
                    outfile.write(f"{code}\n")
                    outfile.write("((CURR++))\n")

                elif language == 'python':
                    context_code = "\n".join(
                        line for line in code_lines if not line.strip().startswith(('print(', 'subprocess.run')))
                    execution_code = "\n".join(
                        line for line in code_lines if line.strip().startswith(('print(', 'subprocess.run')))

                    if context_code:
                        outfile.write(f"echo \"{context_code}\" >> \"$CONTEXT_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"try:\" > \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("sed 's/^/    /' \"$CONTEXT_PYTHON_SCRIPT\" >> \"$EXEC_PYTHON_SCRIPT\"\n")

                    if execution_code:
                        outfile.write(f"echo \"{execution_code}\" | sed 's/^/    /' >> \"$EXEC_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"except Exception as e:\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    print(e)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    import sys; sys.exit(1)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("$VENV_PYTHON \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("((CURR++))\n")

                else:
                    continue

                outfile.write("log_data \"$CURR\"\n")

                outfile.write("if [ \"$CURR\" -eq \"$last_count\" ]; then\n")
                outfile.write("    EVAL=\"null\"\n")
                outfile.write("else\n")
                outfile.write("    EVAL=$((CURR + 1))\n")
                outfile.write("fi\n")

                outfile.write("update_eval_build \"$CURR\" \"$EVAL\"\n\n")

        # logger.info("Temp script generated successfully at %s", temp_sh_path)

    except Exception as e:
        # logger.error("An error occurred while creating temp script: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    custom_sh_file = Path("custom.sh")
    temp_sh_file = Path("temp.sh")

    create_temp_sh(custom_sh_file, temp_sh_file, use_euxo=False)
