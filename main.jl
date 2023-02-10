using Statistics
using Plots
using Base.Threads
include("my_dependencies.jl")

# Data injection
moviesDataFrame, ratingsDataFrame = loadDataSlim("ml-latest-small")

numberOfUsers = length(unique(ratingsDataFrame[:, 1])) # Number of unique "userId" values in ratingsDataFrame
numberOfMovies = size(moviesDataFrame, 1) # Number of rows in moviesDataFrame

# Print data set statistics
#printStatistics()

# Training and test set splitting
testSetSize = 0.10
validationSetSize = 0.10
trainingAndValidationDataFrame, testDataFrame = kFoldSplit(ratingsDataFrame, testSetSize, 0)
testURM = buildURM(testDataFrame, numberOfUsers, numberOfMovies)

# Hyperparameters
similarityMetric = newMetric
aggregationMethod = averageAggregation
errorFunction = meanAbsoluteError
knnMin = 1
knnMax = 2
knnStep = 1
numberOfFolds = 3

println("Training hyperparameters...")
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
    println("- Running for neighborhood size k=$k")
    kFoldErrors = []
    for kFoldIndex = 0:numberOfFolds-1

        println("\t- Iteration $(kFoldIndex + 1)/$numberOfFolds")

        # Training and validation set splitting
        trainingDataFrame, validationDataFrame = kFoldSplit(trainingAndValidationDataFrame, validationSetSize, kFoldIndex)
        
        # Building the URM
        trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)
        validationURM = buildURM(validationDataFrame, numberOfUsers, numberOfMovies)
        urmDensity = getUrmDensityPercentage(trainingURM, trainingDataFrame)

        println("\t\t- URM density: $urmDensity%")

        # Compute validation error
        foldError = computeModelError(trainingURM, validationDataFrame, validationURM, aggregationMethod, k, similarityMetric, errorFunction)
        println("\t\t- Validation error: $foldError")

        push!(kFoldErrors, foldError)

        GC.gc(true) # Explicit call to the garbage collector to make sure no memory is leaked
    end

    # Compute validation error as the average of validation errors on each folds
    avgValError = mean(kFoldErrors)
    stdDevError = std(kFoldErrors)

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
println(" ✓ Neighborhood size k = $bestNeighborhoodSize")

println("\nEvaluating performance on test set...")

# Building the URM
trainingURM = buildURM(trainingAndValidationDataFrame, numberOfUsers, numberOfMovies)

# Printing info
printInfo(trainingURM)
urmDensity = getUrmDensityPercentage(trainingURM, trainingAndValidationDataFrame)
println(" # URM density is $urmDensity%")

# Compute model MAE on the Test Set
mae = computeModelError(trainingURM, testDataFrame, testURM, aggregationMethod, bestNeighborhoodSize, similarityMetric, errorFunction)
println("MAE on test set is $mae")