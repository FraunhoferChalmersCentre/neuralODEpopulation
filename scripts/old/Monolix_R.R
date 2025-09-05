library(lixoftConnectors)
# Load necessary library
library(dplyr)

# Connect R to Monolix
initializeLixoftConnectors(software = "monolix")


loadProject("C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/monolix/FirstOrder.mlxtran")
# Change dataset path






library(dplyr)


n_files1<- 41





n_files2 <- 0


# change this to the number of files you have

# Initialize vectors to store average metrics
n_files=n_files1+n_files2
avg_mse_vector <- numeric(n_files)
avg_r2_vector <- numeric(n_files)

median_mse_vector <- numeric(n_files)
median_r2_vector <- numeric(n_files)

test_data_vector=numeric(0)
files=0:(n_files1-1) #50:(n_files2-1))

# Loop over files
for(i in files){
  BaseData <- getData()
  
  setData(
    file_path <- paste0("C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/run1/monolix_data_", i, ".csv"),
    BaseData$headerTypes,
    BaseData$observationTypes
  )
  
  runPopulationParameterEstimation()
  computeChartsData()
  setPreferences("exportchartsdata"=TRUE)
  
  # Construct file path for the current iteration
  file_path <- "C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/monolix/FirstOrder/ChartsData/ObservationsVsPredictions/DV_obsVsPred.txt"
  
  # Read data
  data <- read.csv(file_path, header = TRUE)
  
  # Check column names
  if(!all(c("ID", "DV", "indivPredMean", "censored") %in% colnames(data))){
    stop(paste("Columns missing in file:", file_path))
  }
  
  # Filter censored data
  censored_data <- data %>% filter(censored == 1)
  
  test_data=data.frame(ID=censored_data$ID,Time=censored_data$time, Obs=censored_data$DV, Pred=censored_data$indivPredMean)
  test_data$Residual=test_data$Obs-test_data$Pred
  test_data$Iteration=i+1
  
  if(nrow(censored_data) == 0){
    stop(paste("No censored data found in file:", file_path))
  }
  
  # Compute per-individual MSE and R²
  per_individual <- censored_data %>%
    group_by(ID) %>%
    summarise(
      mse = mean((DV - indivPredMean)^2, na.rm = TRUE),
      r2 = 1 - sum((DV - indivPredMean)^2, na.rm = TRUE) / sum((DV - mean(DV, na.rm = TRUE))^2, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Compute average metrics for this file
  average_metrics <- per_individual %>%
    summarise(
      avg_mse = mean(mse, na.rm = TRUE),
      avg_r2 = mean(r2, na.rm = TRUE),
      median_mse = median(mse, na.rm = TRUE),
      median_r2 = median(r2, na.rm = TRUE)
    )
  
  # Store in vectors
  avg_mse_vector[i+1] <- average_metrics$avg_mse
  avg_r2_vector[i+1] <- average_metrics$avg_r2
  
  median_mse_vector[i+1] <- average_metrics$median_mse
  median_r2_vector[i+1] <- average_metrics$median_r2
  test_data_vector=rbind(test_data_vector,test_data)
  
}

# Compute overall mean and standard deviation across iterations
overall_mean_mse_mean <- mean(avg_mse_vector, na.rm = TRUE)
overall_mean_mse_sd   <- sd(avg_mse_vector, na.rm = TRUE)
overall_mean_r2_mean  <- mean(avg_r2_vector, na.rm = TRUE)
overall_mean_r2_sd    <- sd(avg_r2_vector, na.rm = TRUE)


overall_median_mse_mean <- mean(median_mse_vector, na.rm = TRUE)
overall_median_mse_sd   <- sd(median_mse_vector, na.rm = TRUE)
overall_median_r2_mean  <- mean(median_r2_vector, na.rm = TRUE)
overall_median_r2_sd    <- sd(median_r2_vector, na.rm = TRUE)


cat("Overall MSE: Mean =", overall_mean_mse_mean, ", SD =", overall_mean_mse_sd, "\n")
cat("Overall R²: Mean =", overall_mean_r2_mean, ", SD =", overall_mean_r2_sd, "\n")

cat("Overall MSE: Median =", overall_median_mse_mean, ", SD =", overall_median_mse_sd, "\n")
cat("Overall R²: Median =", overall_median_r2_mean, ", SD =", overall_median_r2_sd, "\n")


results <- data.frame(
  Metric = c("MSE", "R2", "MSE", "R2"),
  Statistic = c("Mean", "Mean", "Median", "Median"),
  Value = c(overall_mean_mse_mean, overall_mean_r2_mean, overall_median_mse_mean, overall_median_r2_mean),
  SD = c(overall_mean_mse_sd, overall_mean_r2_sd, overall_median_mse_sd, overall_median_r2_sd)
)

# Define the full file path
file_path <- "C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/run1/metrics_SAEM_First_Order.csv"

# Export as CSV
write.csv(results, file_path, row.names = FALSE)

# Define the full file path
file_path2 <- "C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/run1/residuals_SAEM_First_Order.csv"

# Export as CSV
write.csv(test_data_vector, file_path2, row.names = FALSE)



print(results)



loadProject("C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/monolix/ZeroOrder.mlxtran")
# Change dataset path






# Initialize vectors to store average metrics
avg_mse_vector <- numeric(n_files)
avg_r2_vector <- numeric(n_files)

median_mse_vector <- numeric(n_files)
median_r2_vector <- numeric(n_files)
test_data_vector=numeric(0)

# Loop over files
for(i in files){
  BaseData <- getData()
  
  setData(
    file_path <- paste0("C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/run1/monolix_data_", i, ".csv"),
    BaseData$headerTypes,
    BaseData$observationTypes
  )
  
  runPopulationParameterEstimation()
  computeChartsData()
  setPreferences("exportchartsdata"=TRUE)
  
  # Construct file path for the current iteration
  file_path <- "C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/monolix/ZeroOrder/ChartsData/ObservationsVsPredictions/DV_obsVsPred.txt"
  
  # Read data
  data <- read.csv(file_path, header = TRUE)
  
  # Check column names
  if(!all(c("ID", "DV", "indivPredMean", "censored") %in% colnames(data))){
    stop(paste("Columns missing in file:", file_path))
  }
  
  # Filter censored data
  censored_data <- data %>% filter(censored == 1)
  
  test_data=data.frame(ID=censored_data$ID,Time=censored_data$time, Obs=censored_data$DV, Pred=censored_data$indivPredMean)
  test_data$Residual=test_data$Obs-test_data$Pred
  test_data$Iteration=i+1
  
  if(nrow(censored_data) == 0){
    stop(paste("No censored data found in file:", file_path))
  }
  
  # Compute per-individual MSE and R²
  per_individual <- censored_data %>%
    group_by(ID) %>%
    summarise(
      mse = mean((DV - indivPredMean)^2, na.rm = TRUE),
      r2 = 1 - sum((DV - indivPredMean)^2, na.rm = TRUE) / sum((DV - mean(DV, na.rm = TRUE))^2, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Compute average metrics for this file
  average_metrics <- per_individual %>%
    summarise(
      avg_mse = mean(mse, na.rm = TRUE),
      avg_r2 = mean(r2, na.rm = TRUE),
      median_mse = median(mse, na.rm = TRUE),
      median_r2 = median(r2, na.rm = TRUE)
    )
  
  # Store in vectors
  avg_mse_vector[i+1] <- average_metrics$avg_mse
  avg_r2_vector[i+1] <- average_metrics$avg_r2
  
  median_mse_vector[i+1] <- average_metrics$median_mse
  median_r2_vector[i+1] <- average_metrics$median_r2
  test_data_vector=rbind(test_data_vector,test_data)
  
}

# Compute overall mean and standard deviation across iterations
overall_mean_mse_mean <- mean(avg_mse_vector, na.rm = TRUE)
overall_mean_mse_sd   <- sd(avg_mse_vector, na.rm = TRUE)
overall_mean_r2_mean  <- mean(avg_r2_vector, na.rm = TRUE)
overall_mean_r2_sd    <- sd(avg_r2_vector, na.rm = TRUE)


overall_median_mse_mean <- mean(median_mse_vector, na.rm = TRUE)
overall_median_mse_sd   <- sd(median_mse_vector, na.rm = TRUE)
overall_median_r2_mean  <- mean(median_r2_vector, na.rm = TRUE)
overall_median_r2_sd    <- sd(median_r2_vector, na.rm = TRUE)


cat("Overall MSE: Mean =", overall_mean_mse_mean, ", SD =", overall_mean_mse_sd, "\n")
cat("Overall R²: Mean =", overall_mean_r2_mean, ", SD =", overall_mean_r2_sd, "\n")

cat("Overall MSE: Median =", overall_median_mse_mean, ", SD =", overall_median_mse_sd, "\n")
cat("Overall R²: Median =", overall_median_r2_mean, ", SD =", overall_median_r2_sd, "\n")


results <- data.frame(
  Metric = c("MSE", "R2", "MSE", "R2"),
  Statistic = c("Mean", "Mean", "Median", "Median"),
  Value = c(overall_mean_mse_mean, overall_mean_r2_mean, overall_median_mse_mean, overall_median_r2_mean),
  SD = c(overall_mean_mse_sd, overall_mean_r2_sd, overall_median_mse_sd, overall_median_r2_sd)
)


print(results)


# Define the full file path
file_path <- "C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/run1/metrics_SAEM_Zero_Order.csv"

# Export as CSV
write.csv(results, file_path, row.names = FALSE)

# Define the full file path
file_path2 <- "C:/Users/Baaz/OneDrive - Fraunhofer-Chalmers Centre/NODE/neuralODEpopulation/results/run1/residuals_SAEM_Zero_Order.csv"

# Export as CSV
write.csv(test_data_vector, file_path2, row.names = FALSE)