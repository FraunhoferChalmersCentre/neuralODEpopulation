library(tidyverse)
library(lixoftConnectors)
library(dplyr)

# Remove variables
rm(list=ls())

# Manually set your working directory

# Initialize Monolix (still absolute path to MonolixSuite)
initializeLixoftConnectors(
  software = "monolix",
  path = "C:/ProgramData/Lixoft/MonolixSuite2023R1"
)

# Define paths relative to working directory
data_folder <- "results/theo/run5"
mlx_file_no_iiv <- "results/monolix/Monolix_NODE_2023.mlxtran"
mlx_file_iiv <- "results/monolix/Monolix_NODE_pop_2023.mlxtran"
output_pop_project <- "results/monolix/THEOPHYLLINE_project_pop_train_newini1.mlxtran"
output_ind_project <- "results/monolix/THEOPHYLLINE_project_ind_train_newini1.mlxtran"
charts_file <- "results/monolix/THEOPHYLLINE_project_ind_train_newini1/ChartsData/ObservationsVsPredictions/DV_obsVSPred.txt"
metrics_file <- file.path(data_folder, "metrics_monolix.csv")
residuals_file <- file.path(data_folder, "residuals.csv")

iteration_metrics <- data.frame(
  iteration = integer(),
  mean_MSE = numeric(),
  median_MSE = numeric(),
  mean_R2 = numeric(),
  median_R2 = numeric()
)

all_residuals <- data.frame(
  iteration = integer(),
  ID = integer(),
  time = numeric(),
  DV = numeric(),
  indivPredMean = numeric(),
  residual = numeric()
)



calc_metrics <- function(df) {
  observed <- df$DV
  predicted <- df$indivPredMode
  residuals <- observed - predicted
  
  mse <- mean(residuals^2)
  
  ss_res <- sum(residuals^2)
  ss_tot <- sum((observed - mean(observed))^2)
  r2 <- ifelse(ss_tot != 0, 1 - ss_res / ss_tot, NA)
  
  return(data.frame(MSE = mse, R2 = r2))
}


# Number of files
n_files <- 22
files <- 12#13:(n_files - 1)

for (i in files) {
  # Load project without IIV
  loadProject(file.path(getwd(), mlx_file_no_iiv))
  
  BaseData <- getData()
  file_path <- file.path(getwd(), data_folder, paste0("monolix_data_", i, ".csv"))
  setData(file_path, BaseData$headerTypes, BaseData$observationTypes)
  saveProject(file.path(getwd(), output_pop_project))
  runScenario()
  
  # Get population estimates
  est_pop_parms <- getEstimatedPopulationParameters()
  
  # Load project with IIV
  loadProject(file.path(getwd(), mlx_file_iiv))
  
  og_ind_ini <- getPopulationParameterInformation()
  
  new_ind_ini <- og_ind_ini
  pop_idx <- grepl("pop", new_ind_ini$name)
  omega_idx <- grepl("omega", new_ind_ini$name)
  
  new_ind_ini$initialValue[pop_idx] <- est_pop_parms    # length must match sum(pop_idx)
  new_ind_ini$initialValue[omega_idx] <- 0.1
  
  
  
  setPopulationParameterInformation(new_ind_ini)
  setData(file_path, BaseData$headerTypes, BaseData$observationTypes)
  saveProject(file.path(getwd(), output_ind_project))
  runScenario()
  
  # Compute charts
  computeChartsData()
  data <- read.csv(file.path(getwd(), charts_file), stringsAsFactors = FALSE)
  
  # Residuals and metrics calculation
  censored_data <- data %>% filter(censored != 0) %>%
    mutate(residual = DV - indivPredMode, iteration = i) %>%
    select(iteration, ID, time, DV, indivPredMode, residual)
  
  censored_data <- censored_data %>%
    mutate(residual = DV - indivPredMode)
  
  metrics <- censored_data %>%
    group_by(ID) %>%
    do(calc_metrics(.))
  
  # Mean and median across individuals
  mean_mse <- mean(metrics$MSE, na.rm = TRUE)
  median_mse <- median(metrics$MSE, na.rm = TRUE)
  mean_r2 <- mean(metrics$R2, na.rm = TRUE)
  median_r2 <- median(metrics$R2, na.rm = TRUE)
  
  # Print results
  cat("Mean MSE:", mean_mse, "\n")
  cat("Median MSE:", median_mse, "\n")
  cat("Mean R2:", mean_r2, "\n")
  cat("Median R2:", median_r2, "\n")
  
  iteration_metrics <- rbind(iteration_metrics, data.frame(
    iteration = i,
    mean_MSE = mean_mse,
    median_MSE = median_mse,
    mean_R2 = mean_r2,
    median_R2 = median_r2
  ))
  
  # Optionally, print progress
  cat("Iteration", i, "processed.\n")
  
  censored_data <- censored_data %>%
    mutate(residual = DV - indivPredMode) %>%
    mutate(iteration = i) %>%
    select(iteration, ID, time, DV, indivPredMode, residual)
  
  # Append to all_residuals
  all_residuals <- rbind(all_residuals, censored_data)

  write.csv(iteration_metrics, metrics_file, row.names = FALSE)
  write.csv(all_residuals, residuals_file, row.names = FALSE)

}
