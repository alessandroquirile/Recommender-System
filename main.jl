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
#printStatistics()

# Training and test set splitting
testSetSize = 0.10
trainingAndValidationDataFrame, testDataFrame = kFoldSplit(ratingsDataFrame, testSetSize, 0)
testURM = buildURM(testDataFrame, numberOfUsers, numberOfMovies)

# Model parameters
similarityMetric = newMetric
aggregationMethod = averageAggregation
errorFunction = meanAbsoluteError

println("Printing model parameters...")
println(" # Similarity metric: $similarityMetric")
println(" # Aggregation method: $aggregationMethod")
println(" # Error function: $errorFunction")
println("")


knnMin = 2
knnMax = 200
knnStep = 50
numberOfKFolds = 1

validationErrors = []

for k in knnMin:knnStep:knnMax # foreach parameter
    println("- Running for neighborhood size k=$k")
    errorSum = 0.0
    for kFoldIndex = 0:numberOfKFolds-1

        println("\t- Running fold number $(kFoldIndex + 1)")

        # Training and validation set splitting
        validationSetSize = 0.10
        trainingDataFrame, validationDataFrame = kFoldSplit(trainingAndValidationDataFrame, validationSetSize, kFoldIndex)
        
        # Building the URM
        trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)
        validationURM = buildURM(validationDataFrame, numberOfUsers, numberOfMovies)

        # Compute validation error
        foldError = computeModelError(trainingURM, validationDataFrame, validationURM, aggregationMethod, k, similarityMetric, errorFunction)
        println("\t\t- Validation error is $foldError")
        errorSum = errorSum + foldError

        GC.gc(true) # Explicit call to the garbage collector to make sure no memory is leaked
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
println("Best parameters:")
println(" # neighborhood size k = $bestK")

println(" - Evaluating performance on test set")
# Building the URM
trainingURM = buildURM(trainingAndValidationDataFrame, numberOfUsers, numberOfMovies)
# Printing info
printInfo(trainingURM)
printDensity(trainingURM, trainingAndValidationDataFrame)
# Compute model error on the Test Set
error = computeModelError(trainingURM, testDataFrame, testURM, aggregationMethod, bestK, similarityMetric, errorFunction)

println("MAE on test set is $error")