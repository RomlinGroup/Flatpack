use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use bollard::Docker;
use serde::Deserialize;
use std::io;

mod container;

#[derive(Deserialize)]
struct ContainerRequest {
    model_name: String,
    version: String,
}

async fn create_container(docker: &Docker, model_name: &str, version: &str) -> Result<(), String> {
    container::create_and_start_container(docker, model_name, version)
        .await
        .map_err(|e| {
            println!("Error: {}", e);
            e.to_string()
        })
}

#[get("/")]
async fn index() -> impl Responder {
    let html = r#"
        <html>
            <head>
                <title>Flatpack AI</title>
            </head>
            <body>
                <label for="container">Select a flatpack:</label>
                <select id="container" name="model_name">
                    <option value="romlin/flatpack-ai-docker">romlin/flatpack-ai-docker</option>
                </select>
                <br>
                <label for="version">Version:</label>
                <input type="text" id="version" name="version" value="0.4">
                <br>
                <button id="create-container-button" type="submit">Assemble</button>
                <br><br>
                <div id="create-container-result"></div>
                
                <script>
                let createContainerButton = document.getElementById("create-container-button");
                let createContainerResult = document.getElementById("create-container-result");

                createContainerButton.onclick = function() {
                    let model_name = document.getElementById("container").value;
                    let version = document.getElementById("version").value;

                    let xhr = new XMLHttpRequest();
                    xhr.open("POST", "/create", true);
                    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
                    xhr.onload = function() {
                        if (xhr.status === 200) {
                            createContainerResult.innerHTML = xhr.responseText;
                        } else {
                            createContainerResult.innerHTML = "Error: " + xhr.responseText;
                        }
                    };
                    xhr.onerror = function() {
                        createContainerResult.innerHTML = "Error: could not reach server";
                    };
                    xhr.send(JSON.stringify({
                        "model_name": model_name,
                        "version": version
                    }));
                };
                </script>
            </body>
        </html>
    "#;
    HttpResponse::Ok().body(html)
}

#[post("/create")]
async fn handle_create_container(
    docker: web::Data<Docker>,
    form: web::Json<ContainerRequest>,
) -> impl Responder {
    match create_container(&docker, &form.model_name, &form.version).await {
        Ok(_) => HttpResponse::Ok().body(format!(
            "Assembling flatpack for model: {}, version: {}",
            form.model_name, form.version
        )),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    }
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    let docker = Docker::connect_with_local_defaults().unwrap();

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(docker.clone()))
            .service(index)
            .service(handle_create_container)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}