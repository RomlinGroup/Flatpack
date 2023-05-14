const testParserButton = document.getElementById("test-parser-button");
const testParserResult = document.getElementById("test-parser-result");
const testParserResultLabel = document.querySelector("label[for='test-parser-result']");

const handleError = (element, error) => {
    element.value = `Error: ${error || 'Could not reach the server'}`;
};

const updateResult = (result) => {
    const timestamp = new Date().toLocaleString();
    testParserResult.value = result.trim();
    testParserResultLabel.textContent = `Result (generated ${timestamp})`;
};

testParserButton.addEventListener("click", async () => {
    const tomlUrl = encodeURIComponent("https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse/custom-gpt/flatpack.toml");

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