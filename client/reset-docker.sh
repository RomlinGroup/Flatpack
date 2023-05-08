#!/bin/bash
# Script to reset the entire local Docker environment

# Print the initial message
echo "🐳 Resetting the entire local Docker environment!"
printf '\n'

# Prompt the user for confirmation
while true; do
  read -r -p "🤔 Are you absolutely sure? (Y/N) " confirm
  case $confirm in
  [Yy]*) break ;;
  [Nn]*) exit ;;
  *) echo "Please answer yes or no." ;;
  esac
done

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
printf '\n'