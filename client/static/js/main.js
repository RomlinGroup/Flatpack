let createContainerButton = document.getElementById("create-container-button");
let createContainerResult = document.getElementById("create-container-result");

createContainerButton.onclick = function () {
    let model_name = document.getElementById("container").value;
    let version = document.getElementById("version").value;

    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/create", true);
    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
    xhr.onload = function () {
        if (xhr.status === 200) {
            createContainerResult.innerHTML = xhr.responseText;
        } else {
            createContainerResult.innerHTML = "Error: " + xhr.responseText;
        }
    };
    xhr.onerror = function () {
        createContainerResult.innerHTML = "Error: could not reach server";
    };
    xhr.send(JSON.stringify({
        "model_name": model_name,
        "version": version
    }));
};

let testParserButton = document.getElementById("test-parser-button");
let testParserResult = document.getElementById("test-parser-result");

testParserButton.onclick = function () {
    let tomlUrl = encodeURIComponent("http://example.com/path/to/file.toml");
    let xhr = new XMLHttpRequest();
    xhr.open("GET", "/test_parser/" + tomlUrl, true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            testParserResult.innerHTML = xhr.responseText;
        } else {
            testParserResult.innerHTML = "Error: " + (xhr.responseText || "Unknown error");
        }
    };
    xhr.onerror = function () {
        testParserResult.innerHTML = "Error: could not reach server";
    };
    xhr.send();
};