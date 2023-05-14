const createContainerButton = document.getElementById("create-container-button");
const createContainerResult = document.getElementById("create-container-result");
const testParserButton = document.getElementById("test-parser-button");
const testParserResult = document.getElementById("test-parser-result");

const handleError = (element, error) => {
    element.innerHTML = `Error: ${error || 'could not reach server'}`;
}

createContainerButton.onclick = async function () {
    const model_name = document.getElementById("container").value;
    const version = document.getElementById("version").value;

    try {
        const response = await fetch('/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=UTF-8'
            },
            body: JSON.stringify({
                model_name,
                version
            })
        });

        createContainerResult.innerHTML = response.ok ? await response.text() : `Error: ${response.statusText}`;
    } catch (error) {
        handleError(createContainerResult, error);
    }
};

testParserButton.onclick = async function () {
    const tomlUrl = encodeURIComponent("https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse/custom-gpt/flatpack.toml");

    try {
        const response = await fetch(`/test_parser/${tomlUrl}`);

        testParserResult.innerHTML = response.ok ? `[DEBUG] ${await response.text()}` : `Error: ${response.statusText || 'Unknown error'}`;
    } catch (error) {
        handleError(testParserResult, error);
    }
};