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
trainingAndValidationDataFrame, testDataFrame = kFoldSplit(ratingsDataFrame, testSetSize, 0)
testURM = buildURM(testDataFrame, numberOfUsers, numberOfMovies)

# Hyperparameters
similarityMetric = newMetric
aggregationMethod = averageAggregation
errorFunction = meanAbsoluteError
knnMin = 2
knnMax = 200
knnStep = 50
numberOfFolds = 1

println("Training hyperparameters...")
println(" # Validation technique: $numberOfFolds-fold cross validation")
println(" # Similarity metric: $similarityMetric")
println(" # Aggregation method: $aggregationMethod")
println(" # Error function: $errorFunction")
println(" # Neighborhood size: [$knnMin:$knnMax]")
println(" # Neighborhood step: $knnStep")
println("")

validationErrors = []

for k in knnMin:knnStep:knnMax # foreach hyperparameter
    println("- Running for neighborhood size k=$k")
    errorsSum = 0.0
    for kFoldIndex = 0:numberOfFolds-1

        println("\t- Iteration $(kFoldIndex + 1)/$numberOfFolds")

        # Training and validation set splitting
        validationSetSize = 0.10
        trainingDataFrame, validationDataFrame = kFoldSplit(trainingAndValidationDataFrame, validationSetSize, kFoldIndex)
        
        # Building the URM
        trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)
        validationURM = buildURM(validationDataFrame, numberOfUsers, numberOfMovies)

        # Compute validation error
        foldError = computeModelError(trainingURM, validationDataFrame, validationURM, aggregationMethod, k, similarityMetric, errorFunction)
        println("\t\t- Validation error: $foldError")
        errorsSum +=  foldError

        GC.gc(true) # Explicit call to the garbage collector to make sure no memory is leaked
    end

    # Compute validation error as the average of validation errors on each folds
    avgValError = errorsSum / numberOfFolds

    # Store the validation error we just computed in validationErrors
    push!(validationErrors, (k, avgValError))
end

println("✓ Trained")

plotValidationHistory(validationErrors)

# Performance evaluation
sort!(validationErrors, by = x -> x[2])
bestNeighborhoodSize = validationErrors[1][1]
println("\nModel selection...")
println(" ✓ Neighborhood size k = $bestNeighborhoodSize")
print("")

println("Evaluating performance on test set...")

# Building the URM
trainingURM = buildURM(trainingAndValidationDataFrame, numberOfUsers, numberOfMovies)

# Printing info
printInfo(trainingURM)
printDensity(trainingURM, trainingAndValidationDataFrame)

# Compute model MAE on the Test Set
mae = computeModelError(trainingURM, testDataFrame, testURM, aggregationMethod, bestNeighborhoodSize, similarityMetric, errorFunction)
println("MAE on test set is $mae")