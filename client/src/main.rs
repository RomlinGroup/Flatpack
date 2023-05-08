use actix_files::Files;
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
async fn index() -> HttpResponse {
    let html = include_str!("../templates/main.html");
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
            .service(Files::new("/static", "static").show_files_listing())
            .service(index)
            .service(handle_create_container)
    })
        .bind("127.0.0.1:8080")?
        .run()
        .await
}