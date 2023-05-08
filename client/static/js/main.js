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