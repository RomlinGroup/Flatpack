const testParserButton = document.getElementById("test-parser-button");
const testParserResult = document.getElementById("test-parser-result");
const downloadButton = document.getElementById("download-button");

const handleError = (element, error) => {
    element.value = `Error: ${error || 'Could not reach the server'}`;
};

const updateResult = (result) => {
    testParserResult.value = result.trim();
    downloadButton.disabled = false;
};

testParserButton.addEventListener("click", async () => {
    const tomlUrl = encodeURIComponent("https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse/nanogpt-shakespeare/flatpack.toml");

    try {
        // Disable the button and show a progress indicator
        testParserButton.disabled = true;
        testParserButton.innerHTML = "Working...";

        const response = await fetch(`/test_parser/${tomlUrl}`);
        const result = response.ok ? await response.text() : `Error: ${response.statusText || 'Unknown error'}`;
        updateResult(result);
    } catch (error) {
        handleError(testParserResult, error);
    } finally {
        // Re-enable the button and remove the progress indicator
        testParserButton.disabled = false;
        testParserButton.innerHTML = "Test Parser";
    }
});

downloadButton.addEventListener("click", () => {
    const dockerfileContent = testParserResult.value;

    if (dockerfileContent.trim() === "") {
        alert("Please generate the Dockerfile first.");
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