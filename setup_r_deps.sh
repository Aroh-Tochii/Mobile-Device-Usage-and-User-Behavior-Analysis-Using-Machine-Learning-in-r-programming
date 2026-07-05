#!/usr/bin/env bash
# Install system libraries required by R packages (Shiny stack, etc.)
set -euo pipefail

echo "Installing system dependencies..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libuv1-dev \
  build-essential \
  libcurl4-openssl-dev \
  libssl-dev \
  libxml2-dev

echo "Installing R packages..."
Rscript "$(dirname "$0")/install_packages.R"

echo "Done. Verify with: Rscript -e \"library(shiny); library(shinydashboard)\""
