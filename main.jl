using Statistics
using Plots
using Base.Threads
using StatsBase
include("my_dependencies.jl")

# Data injection
moviesDataFrame, ratingsDataFrame = loadData("ml-latest-small")

numberOfUsers = length(unique(ratingsDataFrame[:, 1])) # Number of unique "userId" values in ratingsDataFrame
numberOfMovies = size(moviesDataFrame, 1) # Number of rows in moviesDataFrame

# Print data set statistics
#### printStatistics()

# Training and test set splitting
testSetSize = 0.10
validationSetSize = 0.10
trainingDataFrame, testDataFrame = datasetSplit(ratingsDataFrame, testSetSize, 0)
testURM = buildURM(testDataFrame, numberOfUsers, numberOfMovies)

# Hyperparameters
similarityMetric = pearsonCorrelation
aggregationMethod = adjustedWeightedSumAggregation
errorFunction = meanAbsoluteError
knnMin = 1
knnMax = 150 # based on the maximum size in paper, scaled according to dataset size
knnStep = 5 # based on the maximum size in paper, scaled according to dataset size
numberOfFolds = 3


println("Training's hyperparameters...")
println(" # Validation technique: $numberOfFolds-fold cross validation")
println(" # Total data is split into $((1-testSetSize)*100)% training and $(testSetSize*100)% test")
println(" # Validation set is $(validationSetSize*100)% of training data")
println(" # Similarity metric: $similarityMetric")
println(" # Aggregation method: $aggregationMethod")
println(" # Error function: $errorFunction")
println(" # Neighborhood size: [$knnMin:$knnMax]")
println(" # Neighborhood step: $knnStep")
println("")


validationErrorsMean = []
validationErrorsStdDev = []

for k in knnMin:knnStep:knnMax # foreach hyperparameter
    println("- Running for neighborhood size k = $k")
    kFoldErrors = []
    for kFoldIndex = 0:numberOfFolds-1

        println("\t- Iteration $(kFoldIndex + 1)/$numberOfFolds")

        # Training and validation set splitting
        modelBuildingDataFrame, validationDataFrame = datasetSplit(trainingDataFrame, validationSetSize, kFoldIndex)
        
        # Building the URM
        modelBuildingURM = buildURM(modelBuildingDataFrame, numberOfUsers, numberOfMovies)
        validationURM = buildURM(validationDataFrame, numberOfUsers, numberOfMovies)
        urmDensity = getDensityPercentage(modelBuildingURM, modelBuildingDataFrame)

        println("\t\t- URM density: $urmDensity%")

        # Compute validation error
        targets, predictions = computePredictions(modelBuildingURM, validationDataFrame, validationURM, aggregationMethod, k, similarityMetric)
        foldError = errorFunction(targets, predictions)
        println("\t\t- Validation error: $foldError")

        push!(kFoldErrors, foldError)

        GC.gc(true) # Explicit call to the garbage collector to make sure no memory is leaked
    end

    # Compute validation error as the average of validation errors on each folds
    avgValError = mean(kFoldErrors)
    stdDevError = std(kFoldErrors)
    println("\t# Avg validation error: $(round(avgValError, digits=3))")
    println("\t# StdDev validation error: $(round(stdDevError, digits=3))")

    # Store the validation error we just computed in validationErrors
    push!(validationErrorsMean, (k, avgValError))
    push!(validationErrorsStdDev, (k, stdDevError))
end

println("✓ Trained")

plotValidationHistory(validationErrorsMean, "Validation error mean")
println("Press a key to continue...")
readline()
plotValidationHistory(validationErrorsStdDev, "Validation error std dev")

# Performance evaluation
sort!(validationErrorsMean, by = x -> x[2])
bestNeighborhoodSize = validationErrorsMean[1][1]
println("\nModel selection...")
println(" ✓ Best neighborhood size k = $bestNeighborhoodSize")

println("\nEvaluating performance on test set...")



# Building the URM
trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)

# Printing info
printInfo(trainingURM)
urmDensity = getDensityPercentage(trainingURM, trainingDataFrame)
println(" # URM density is $urmDensity%")

# Compute model MAE on the Test Set
targets, predictions = computePredictions(trainingURM, testDataFrame, testURM, aggregationMethod, bestNeighborhoodSize, similarityMetric)
mae = errorFunction(targets, predictions)
_precision, recall = computePrecisionAndRecall(targets, predictions)
fMeasure = (2 * _precision * recall) / (_precision + recall)
perfectPredictions = computeNumberOfPerfectPredictions(targets, predictions)

# Print Test Set results
println("MAE on test set is $mae")
println("Precision on test set is $_precision")
println("Recall on test set is $recall")
println("F-Measure is $fMeasure")
println("Perfect predictions: $perfectPredictions/$(length(targets))")