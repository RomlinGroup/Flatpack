use actix_files::Files;
use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use bollard::Docker;
use std::fs::File;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use structopt::StructOpt;
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

#[derive(StructOpt)]
#[structopt(name = "flatpack", about = "flatpack.ai CLI")]
enum Opt {
    Parse {
        /// TOML file path
        #[structopt(parse(from_os_str))]
        path: std::path::PathBuf,
    },
    RunServer,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();

    match opt {
        Opt::Parse { path } => {
            let file_path = path.to_str().unwrap_or_default().to_string();
            match parser::parse_toml_to_dockerfile(&file_path).await {
                Ok(dockerfile) => {
                    // Write the Containerfile to a local file
                    let mut file = File::create("Containerfile").expect("Could not create file");
                    file.write_all(dockerfile.as_bytes()).expect("Could not write to file");

                    // Print the generated Dockerfile
                    println!("Dockerfile generated:\n{}", dockerfile);

                    return Ok(());
                }
                Err(e) => {
                    eprintln!("Error when parsing TOML: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Opt::RunServer => {
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
    }
}