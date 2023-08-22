use reqwest;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;

#[derive(Deserialize)]
pub struct Config {
    dataset: Option<Vec<BTreeMap<String, String>>>,
    directories: Option<BTreeMap<String, String>>,
    environment: BTreeMap<String, String>,
    file: Option<Vec<BTreeMap<String, String>>>,
    git: Vec<BTreeMap<String, String>>,
    packages: Option<BTreeMap<String, BTreeMap<String, String>>>,
    run: Option<Vec<BTreeMap<String, String>>>,
    #[allow(dead_code)]
    version: String,
}

// BEGIN Bash
pub async fn parse_toml_to_pyenv_script(url: &str) -> Result<String, Box<dyn Error>> {
    let response = reqwest::get(url).await?;
    if !response.status().is_success() {
        return Err(format!("Failed to get file from URL: server responded with status code {}", response.status()).into());
    }
    let res = response.text().await?;
    let config: Config = toml::from_str(&res)?;
    let model_name = config.environment.get("model_name").ok_or("Missing model_name in flatpack.toml")?;
    let mut script = String::new();

    script.push_str("#!/bin/bash\n");

    script.push_str("if [[ \"${COLAB_GPU}\" == \"1\" ]]; then\n");
    script.push_str("  echo \"Running in Google Colab environment\"\n");
    script.push_str("  IS_COLAB=1\n");
    script.push_str("else\n");
    script.push_str("  echo \"Not running in Google Colab environment\"\n");
    script.push_str("  IS_COLAB=0\n");
    script.push_str("fi\n");

    script.push_str("if [[ $IS_COLAB -eq 0 ]]; then\n");

    script.push_str(" if ! command -v pyenv >/dev/null; then\n");
    script.push_str("   echo \"pyenv not found. Please install pyenv.\"\n");
    script.push_str("   exit 1\n");
    script.push_str(" fi\n");

    script.push_str(" if ! command -v wget >/dev/null; then\n");
    script.push_str("   echo \"wget not found. Please install wget.\"\n");
    script.push_str("   exit 1\n");
    script.push_str(" fi\n");

    script.push_str(" if ! command -v git >/dev/null; then\n");
    script.push_str("   echo \"git not found. Please install git.\"\n");
    script.push_str("   exit 1\n");
    script.push_str(" fi\n");

    script.push_str(" export PYENV_ROOT=\"$HOME/.pyenv\"\n");
    script.push_str(" export PATH=\"$PYENV_ROOT/bin:$PATH\"\n");
    script.push_str(" if command -v pyenv 1>/dev/null 2>&1; then\n");
    script.push_str("   eval \"$(pyenv init -)\"\n");
    script.push_str("   eval \"$(pyenv virtualenv-init -)\"\n");
    script.push_str(" fi\n");

    script.push_str("fi\n");

    // Create a new project directory
    script.push_str(&format!("mkdir -p ./{}\n", model_name));

    // Create directories
    if let Some(directories_map) = &config.directories {
        for (_directory_name, directory_path) in directories_map {
            let formatted_directory_path = directory_path.trim_start_matches('/');
            let without_home_content = formatted_directory_path.trim_start_matches("home/content/");
            script.push_str(&format!("mkdir -p ./{}/{}\n", model_name, without_home_content));
        }
    } else {
        script.push_str("# Found no directories, proceeding without it.\n");
    }

    // Set environment variables
    for (key, value) in &config.environment {
        script.push_str(&format!("export {}={}\n", key, value.replace("/home/content/", &format!("./{}/", model_name))));
    }

    // Create a new pyenv environment and activate it
    let version = "3.11.3";
    let env_name = "myenv";
    script.push_str(" if [[ $IS_COLAB -eq 0 ]]; then\n");
    script.push_str(&format!(" if ! pyenv versions | grep -q {0}; then\n  pyenv install {0}\nfi\n", version));
    script.push_str(&format!(" if ! pyenv virtualenvs | grep -q {0}; then\n  pyenv virtualenv {1} {0}\nfi\n", env_name, version));
    script.push_str(&format!(" pyenv activate {}\n", env_name));
    script.push_str("fi\n");

    // Install Python packages
    if let Some(packages) = &config.packages {
        if let Some(python_packages) = packages.get("python") {
            let package_list: Vec<String> = python_packages
                .iter()
                .map(|(package, version)| {
                    if version == "*" || version.is_empty() {
                        format!("{}", package)  // If version is not specified or "*", get the latest version
                    } else {
                        format!("{}=={}", package, version)  // If version is specified, get that version
                    }
                })
                .collect();
            script.push_str(&format!(
                "python -m pip install {}\n",
                package_list.join(" ")
            ));
        }
    }

    // Git repositories
    for git in &config.git {
        if let (Some(from_source), Some(to_destination), Some(branch)) = (git.get("from_source"), git.get("to_destination"), git.get("branch")) {
            let repo_path = format!("./{}/{}", model_name, to_destination.replace("/home/content/", ""));
            script.push_str(&format!("echo 'Cloning repository from: {}'\n", from_source));
            script.push_str(&format!("git clone -b {} {} {}\n", branch, from_source, repo_path));
            script.push_str(&format!("if [ -f {}/requirements.txt ]; then\n  echo 'Found requirements.txt, installing dependencies...'\n  cd {} || exit\n  python -m pip install -r requirements.txt\n  cd - || exit\nelse\n  echo 'No requirements.txt found.'\nfi\n", repo_path, repo_path));
        }
    }

    // Download datasets and files
    if let Some(dataset_vec) = &config.dataset {
        for dataset in dataset_vec {
            if let (Some(from_source), Some(to_destination)) = (dataset.get("from_source"), dataset.get("to_destination")) {
                script.push_str(&format!("wget {} -O ./{}/{}\n", from_source, model_name, to_destination.replace("/home/content/", "")));
            }
        }
    } else {
        script.push_str("# Found no datasets, proceeding without them.\n");
    }

    // Download files
    if let Some(file_vec) = &config.file {
        for file in file_vec.iter() {
            if let (Some(from_source), Some(to_destination)) = (file.get("from_source"), file.get("to_destination")) {
                script.push_str(&format!("wget {} -O ./{}/{}\n", from_source, model_name, to_destination.replace("/home/content/", "")));
            }
        }
    } else {
        script.push_str("# Found no files, proceeding without them.\n");
    }

    // RUN commands
    if let Some(run_vec) = &config.run {
        for run in run_vec {
            if let (Some(command), Some(args)) = (run.get("command"), run.get("args")) {
                // replace "/home/content/" with "./{model_name}/"
                let replaced_args = args.replace("/home/content/", &format!("./{}/", model_name));
                script.push_str(&format!("{} {}\n", command, replaced_args));
            }
        }
    } else {
        script.push_str("# Found no run commands, proceeding without them.\n");
    }

    Ok(script)
}
// END Bash