const testParserButton = document.getElementById("test-parser-button");
const testParserResult = document.getElementById("test-parser-result");
const testParserPyenvResult = document.getElementById("test-parser-pyenv-result");
const downloadButton = document.getElementById("download-button");
const flatpackSelector = document.getElementById("flatpack-selector");

const handleError = (element, error) => {
    element.value = `Error: ${error || 'Could not reach the server'}`;
};

const updateResult = (result) => {
    testParserResult.value = result.trim();
    // Only enable the Download button if the result doesn't start with "Error"
    downloadButton.disabled = result.trim().startsWith('Error');

    // Test
    testParserPyenvResult.value = "Hello, World!"
};

testParserButton.addEventListener("click", async () => {
    const selectedFlatpack = flatpackSelector.value;

    // If no flatpack is selected, show a message in the textarea and stop the execution of the function
    if (!selectedFlatpack) {
        testParserResult.value = 'Please select a flatpack.';
        downloadButton.disabled = true;  // Disable the Download button
        return;
    }

    const tomlUrl = encodeURIComponent(`https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse/${selectedFlatpack}/flatpack.toml`);

    try {
        testParserButton.disabled = true;
        testParserButton.innerHTML = "Working...";

        const response = await fetch(`/test_parser/${tomlUrl}`);
        const result = response.ok ? await response.text() : `Error: ${response.statusText || 'Unknown error'}`;
        updateResult(result);
    } catch (error) {
        handleError(testParserResult, error);
    } finally {
        testParserButton.disabled = false;
        testParserButton.innerHTML = "Test Parser";
    }
});

downloadButton.addEventListener("click", () => {
    const dockerfileContent = testParserResult.value;

    if (dockerfileContent.trim() === "") {
        alert("Please generate the Containerfile first.");
        return;
    }

    const blob = new Blob([dockerfileContent], {type: "text/plain"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "Containerfile";
    anchor.click();
    URL.revokeObjectURL(url);
});

// Reset the dropdown on page load
window.addEventListener('load', function () {
    const dropdown = document.getElementById('flatpack-selector');
    dropdown.selectedIndex = 0; // Set the selected index to the first option (index 0)
});