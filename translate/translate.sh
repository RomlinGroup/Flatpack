#!/bin/bash

# The API endpoint
API="http://localhost:5000/translate"

# The source language
SOURCE="en"

# The target language
TARGET="sv"

# The JSON file to translate
JSON_FILE="en.json"

# The output JSON file
OUTPUT_FILE="sv.json"

# Temporary JSON file
TEMP_FILE="temp.json"

# The URL of the JSON file to download
JSON_URL="https://raw.githubusercontent.com/romlingroup/OpenAlpaca/main/openalpaca.json"

# Download the JSON file
echo "🚀 Starting download of JSON file..."
curl -o "$JSON_FILE" "$JSON_URL"
echo "✅ Download completed."

# Empty the output and temporary files
: >"$OUTPUT_FILE"
: >"$TEMP_FILE"

# Total number of items to translate
TOTAL_ITEMS=$(jq length "$JSON_FILE")

# Counter
COUNTER=0

# Read the JSON file as a whole and iterate over the individual objects
echo "🚀 Starting translation process..."
jq -c '.[]' "$JSON_FILE" | while IFS= read -r object; do
  let COUNTER=COUNTER+1
  echo -ne "[$COUNTER/$TOTAL_ITEMS] Translating item...\r"

  INSTRUCTION=$(echo "$object" | jq -r '.instruction')
  INPUT=$(echo "$object" | jq -r '.input')
  OUTPUT=$(echo "$object" | jq -r '.output')

  TRANSLATED_INSTRUCTION=$(curl -s -X POST -H 'Content-Type: application/x-www-form-urlencoded' -d "q=$INSTRUCTION&source=$SOURCE&target=$TARGET" "$API" | jq -r '.translatedText')
  TRANSLATED_INPUT=$(curl -s -X POST -H 'Content-Type: application/x-www-form-urlencoded' -d "q=$INPUT&source=$SOURCE&target=$TARGET" "$API" | jq -r '.translatedText')
  TRANSLATED_OUTPUT=$(curl -s -X POST -H 'Content-Type: application/x-www-form-urlencoded' -d "q=$OUTPUT&source=$SOURCE&target=$TARGET" "$API" | jq -r '.translatedText')

  # Write the translated fields back to the JSON object and add it to the temporary file
  echo "$object" | jq --arg TI "$TRANSLATED_INSTRUCTION" --arg TIN "$TRANSLATED_INPUT" --arg TO "$TRANSLATED_OUTPUT" '.instruction = $TI | .input = $TIN | .output = $TO' | jq -c '.' >>"$TEMP_FILE"
done

echo -e "\n🔄 Finalizing translation..."

# Construct the JSON array and save to the output file
jq -s '.' "$TEMP_FILE" >"$OUTPUT_FILE"

echo "🏁 Finished translation process. Validating JSON..."

# Validate the output JSON
if jq . "$OUTPUT_FILE" >/dev/null 2>&1; then
  echo "✅ JSON is valid"
else
  echo "❌ JSON is invalid"
fi

# Cleanup
rm "$TEMP_FILE"
