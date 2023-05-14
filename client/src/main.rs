use actix_files::Files;
use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use bollard::Docker;
use std::io;

mod parser;

#[get("/")]
async fn index() -> HttpResponse {
    let html = include_str!("../templates/main.html");
    HttpResponse::Ok().body(html)
}

#[get("/test_parser/{path}")]
async fn test_parser(path: web::Path<String>) -> impl Responder {
    let url = path.into_inner();
    match parser::parse_toml_to_dockerfile(&url).await {
        Ok(dockerfile) => HttpResponse::Ok().content_type("text/plain").body(dockerfile),
        Err(e) => HttpResponse::InternalServerError().content_type("text/plain").body(format!("Error when parsing TOML: {}", e)),
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
            .service(test_parser)
    })
        .bind("127.0.0.1:8080")?
        .run()
        .await
}