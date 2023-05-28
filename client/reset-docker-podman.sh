#!/bin/bash
# Script to reset the local Docker and Podman environments

# Function to reset Docker
reset_docker() {
  echo "🐳 Resetting the entire local Docker environment!"
  # Stop and remove all Docker containers
  containers=$(docker ps -a -q)
  if [ -n "$containers" ]; then
    echo "🧹 Wiping Docker containers."
    docker rm -f $containers
  else
    echo "🚀 No Docker containers to remove."
  fi
  # Remove all Docker images
  images=$(docker images -a -q)
  if [ -n "$images" ]; then
    echo "🗑️ Removing Docker images."
    docker rmi -f $images
  else
    echo "🚀 No Docker images to remove."
  fi
  # Prune unused Docker objects and networks
  echo "♻️ Cleaning up unused Docker objects and networks."
  docker system prune --all --force
  docker network prune --force
}

# Function to reset Podman
reset_podman() {
  echo "🦭 Resetting the entire local Podman environment!"
  # Stop and remove all Podman containers
  containers=$(podman ps -a -q)
  if [ -n "$containers" ]; then
    echo "🧹 Wiping Podman containers."
    podman rm -f $containers
  else
    echo "🚀 No Podman containers to remove."
  fi
  # Remove all Podman images
  images=$(podman images -a -q)
  if [ -n "$images" ]; then
    echo "🗑️ Removing Podman images."
    podman rmi -f $images
  else
    echo "🚀 No Podman images to remove."
  fi
  # Prune unused Podman objects and networks
  echo "♻️ Cleaning up unused Podman objects and networks."
  podman volume prune -f
  podman network prune -f
}

# Prompt the user for confirmation
while true; do
  read -r -p "🤔 Are you absolutely sure? (Y/N) " confirm
  case $confirm in
  [Yy]*) break ;;
  [Nn]*) exit ;;
  *) echo "Please answer yes or no." ;;
  esac
done

# Check if Docker is installed and running
if command -v docker &>/dev/null; then
  if ! docker ps >/dev/null 2>&1; then
    echo "Docker is not running."
  else
    reset_docker
  fi
else
  echo "Docker could not be found. Skipping Docker cleanup."
fi

# Check if Podman is installed. Podman is daemonless, so there's no background service to check if it's running.
if command -v podman &>/dev/null; then
  reset_podman
else
  echo "Podman could not be found. Skipping Podman cleanup."
fi

printf '\n'
