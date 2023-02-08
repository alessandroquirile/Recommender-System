using Statistics
using Plots
using Base.Threads
include("dataset_injection.jl")
include("utils.jl")
include("rs.jl")
include("analytics.jl")
include("metrics.jl")
include("dataset_split.jl")
include("aggregation_methods.jl")
include("performance_evaluation.jl")

# Data injection
moviesDataFrame, ratingsDataFrame = loadDataSlim("ml-latest-small")

numberOfUsers = length(unique(ratingsDataFrame[:, 1])) # Number of unique "userId" values in ratingsDataFrame
numberOfMovies = size(moviesDataFrame, 1) # Number of rows in moviesDataFrame

# Print data set statistics
printStatistics()

# Training and test set splitting
testSetSize = 0.10
trainingAndValidationDataFrame, testDataFrame = kFoldSplit(ratingsDataFrame, numberOfUsers, numberOfMovies, testSetSize, 0)
testURM = buildURM(testDataFrame, numberOfUsers, numberOfMovies)

# Model parameters
similarityMetric = newMetric
aggregationMethod = averageAggregation
errorFunction = meanAbsoluteError



knnMin = 2
knnMax = 10
knnStep = 2
numberOfKFolds = 10

validationErrors = []

for k in knnMin:knnStep:knnMax # foreach parameter
    errorSum = 0.0
    for kFoldIndex = 0:numberOfKFolds-1

        println("Running fold number $kFoldIndex")

        # Training and validation set splitting
        validationSetSize = 0.10
        trainingDataFrame, validationDataFrame = kFoldSplit(trainingAndValidationDataFrame, numberOfUsers, numberOfMovies, validationSetSize, kFoldIndex)

        # Building the URM
        trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)

        # Printing info
        printInfo(trainingURM)
        printDensity(trainingURM, trainingDataFrame)

        # Compute validation error
        foldError = computeModelError(trainingURM, validationDataFrame, aggregationMethod, k, similarityMetric, errorFunction)
        println("Validation error for k=$k, fold #$(kFoldIndex+1) is $foldError")
        errorSum = errorSum + foldError
    end

    # Compute validation error as the average of validation errors on each folds
    validationErrorMean = errorSum / numberOfKFolds
    # Store the validation error we just computed in validationErrors
    push!(validationErrors, (k, validationErrorMean))
end

plotValidationHistory(validationErrors)

# Performance evaluation
sort!(validationErrors, by = x -> x[2])
bestK = validationErrors[1][1]
println("Best k is $bestK")

# Building the URM
trainingURM = buildURM(trainingAndValidationDataFrame, numberOfUsers, numberOfMovies)
error = computeModelError(trainingURM, testDataFrame, aggregationMethod, bestK, similarityMetric, errorFunction)

println("MAE on test set is $error")