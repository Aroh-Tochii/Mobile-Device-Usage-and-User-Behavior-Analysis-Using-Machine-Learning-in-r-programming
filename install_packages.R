# Install R packages for the Mobile Phone Usage project.
# Run setup_r_deps.sh first if fs/sass/shiny fail to build.

user_lib <- Sys.getenv(
  "R_LIBS_USER",
  unset = file.path(Sys.getenv("HOME"), "R", paste0("x86_64-pc-linux-gnu", substr(getRversion(), 1, 3)))
)
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- c(
  "e1071", "caret", "randomForest", "nnet", "caTools",
  "ggplot2", "dplyr", "tidyr",
  "shiny", "shinydashboard"
)

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) == 0) {
  message("All packages already installed.")
} else {
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing)
}
