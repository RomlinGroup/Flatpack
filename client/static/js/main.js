const testParserButton = document.getElementById("test-parser-button");
const testParserResult = document.getElementById("test-parser-result");
const testParserPyenvResult = document.getElementById("test-parser-pyenv-result");
const downloadButton = document.getElementById("download-button");
const downloadPyenvButton = document.getElementById("download-pyenv-button");
const flatpackSelector = document.getElementById("flatpack-selector");

const handleError = (element, error) => {
    element.value = `Error: ${error || 'Could not reach the server'}`;
};

const updateContainerfileResult = async (tomlUrl) => {
    const response = await fetch(`/test_parser/${tomlUrl}`);
    const result = response.ok ? await response.text() : `Error: ${response.statusText || 'Unknown error'}`;
    testParserResult.value = result.trim();
    downloadButton.disabled = result.trim().startsWith('Error');
};

const updatePyenvResult = async (tomlUrl) => {
    const response = await fetch(`/test_pyenv_parser/${tomlUrl}`);
    const result = response.ok ? await response.text() : `Error: ${response.statusText || 'Unknown error'}`;
    testParserPyenvResult.value = result.trim();
    downloadPyenvButton.disabled = result.trim().startsWith('Error'); // Enable/Disable the download Pyenv button based on the result
};

testParserButton.addEventListener("click", async () => {
    const selectedFlatpack = flatpackSelector.value;
    if (!selectedFlatpack) {
        testParserResult.value = 'Please select a flatpack.';
        testParserPyenvResult.value = 'Please select a flatpack.';
        downloadButton.disabled = true;
        downloadPyenvButton.disabled = true; // New Pyenv download button
        return;
    }

    const tomlUrl = encodeURIComponent(`https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse/${selectedFlatpack}/flatpack.toml`);

    testParserButton.disabled = true;
    testParserButton.innerHTML = "Working...";
    try {
        await Promise.all([updateContainerfileResult(tomlUrl), updatePyenvResult(tomlUrl)]);
    } catch (error) {
        handleError(testParserResult, error);
        handleError(testParserPyenvResult, error);
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

// New event listener for Pyenv download button
downloadPyenvButton.addEventListener("click", () => {
    const pyenvContent = testParserPyenvResult.value;
    if (pyenvContent.trim() === "") {
        alert("Please generate the Pyenv script first.");
        return;
    }

    const blob = new Blob([pyenvContent], {type: "text/plain"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "flatpack.sh";
    anchor.click();
    URL.revokeObjectURL(url);
});

// Reset the dropdown and disable buttons on page load
window.addEventListener('load', function () {
    const dropdown = document.getElementById('flatpack-selector');
    dropdown.selectedIndex = 0; // Set the selected index to the first option (index 0)
    downloadButton.disabled = true; // Disable download button on page load
    downloadPyenvButton.disabled = true; // Disable download Pyenv button on page load
});