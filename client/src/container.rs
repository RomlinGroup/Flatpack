use bollard::container::{Config, CreateContainerOptions};
use bollard::Docker;
use futures_util::stream::TryStreamExt;
use std::collections::HashMap;

pub async fn create_and_start_container(
    docker: &Docker,
    model_name: &str,
    version: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let image_name = format!("{}:{}", model_name, version);

    // Check if image exists on system
    let image_exists = docker.inspect_image(&image_name).await.is_ok();

    // If image doesn't exist, download it
    if !image_exists {
        let options = Some(bollard::image::CreateImageOptions {
            from_image: model_name,
            tag: version,
            ..Default::default()
        });

        let mut stream = docker.create_image(options, None, None);

        while let Some(result) = stream.try_next().await? {
            println!("Image layer pulled: {}", result.id.unwrap_or_default());
        }
    }

    let mut exposed_ports = HashMap::new();
    exposed_ports.insert("8080/tcp".to_owned(), HashMap::new());

    let container_config = Config {
        image: Some(image_name),
        exposed_ports: Some(exposed_ports),
        ..Default::default()
    };

    let container_name = format!("{}_{}", model_name, version).replace(".", "-").replace("/", "-");
    let create_options = CreateContainerOptions {
        name: &container_name,
    };

    let container = docker
        .create_container(Some(create_options), container_config)
        .await?;

    docker.start_container::<String>(&container.id, None).await?;

    Ok(())
}