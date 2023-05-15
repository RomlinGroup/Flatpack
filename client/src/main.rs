use actix_files::Files;
use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use bollard::Docker;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};
use tera::{Tera, Context};

mod parser;

#[get("/")]
async fn index() -> HttpResponse {
    let tera = Tera::new("templates/**/*").unwrap();

    // Get the current timestamp
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let timestamp = since_the_epoch.as_secs();

    let mut context = Context::new();
    context.insert("timestamp", &timestamp.to_string());

    let html = tera.render("main.html", &context).unwrap();

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