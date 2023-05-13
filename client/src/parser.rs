use dockerfile_parser::Dockerfile;
use reqwest;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Config {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    license: String,
    base_image: String,
    environment: HashMap<String, String>,
    port: Vec<HashMap<String, u16>>,
    directories: HashMap<String, String>,
    packages: HashMap<String, HashMap<String, String>>,
    dataset: Vec<HashMap<String, String>>,
    file: Vec<HashMap<String, String>>,
}

pub async fn parse_toml_to_dockerfile(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;

    if !response.status().is_success() {
        return Err(format!("Failed to get file from URL: server responded with status code {}", response.status()).into());
    }

    let res = response.text().await?;
    let config: Config = toml::from_str(&res)?;

    // Start building the Dockerfile string
    let mut dockerfile = String::new();

    // Base Image
    dockerfile.push_str(&format!("FROM {}\n\n", config.base_image));

    // Environment Variables
    for (key, value) in config.environment.iter() {
        dockerfile.push_str(&format!("ENV {}={}\n", key, value));
    }

    // Directories
    for (_, value) in config.directories.iter() {
        dockerfile.push_str(&format!("RUN mkdir -p {}\n", value));
    }

    // Packages
    dockerfile.push_str("\n# Install packages\n");
    for (package_type, packages) in config.packages.iter() {
        if package_type == "unix" {
            for (package, _) in packages {
                dockerfile.push_str(&format!("RUN apt-get install -y {}\n", package));
            }
        }
        if package_type == "python" {
            for (package, _) in packages {
                dockerfile.push_str(&format!("RUN pip install {}\n", package));
            }
        }
    }

    // Ports
    for port in config.port.iter() {
        if let Some(internal) = port.get("internal") {
            dockerfile.push_str(&format!("EXPOSE {}\n", internal));
        }
    }

    // Dataset and file downloads
    dockerfile.push_str("\n# Download datasets and files\n");
    for dataset in config.dataset.iter() {
        if let (Some(from_source), Some(to_destination)) = (dataset.get("from_source"), dataset.get("to_destination")) {
            dockerfile.push_str(&format!("RUN wget {} -O {}\n", from_source, to_destination));
        }
    }
    for file in config.file.iter() {
        if let (Some(from_source), Some(to_destination)) = (file.get("from_source"), file.get("to_destination")) {
            dockerfile.push_str(&format!("RUN wget {} -O {}\n", from_source, to_destination));
        }
    }

    // Validate Dockerfile syntax
    let _ = Dockerfile::parse(&dockerfile)?;

    Ok(dockerfile)
}