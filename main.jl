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

# Hyperparameters
similarityMetric = newMetric
aggregationMethod = averageAggregation
errorFunction = meanAbsoluteError
knnMin = 1
knnMax = 150 # based on the maximum size in paper, scaled according to dataset size
knnStep = 5 # based on the maximum size in paper, scaled according to dataset size
numberOfFolds = 5

println("Training's hyperparameters...")
println(" # Validation technique: $numberOfFolds-fold cross validation")
println(" # Total data is split into $((1-testSetSize)*100)% training and $(testSetSize*100)% test")
println(" # Similarity metric: $similarityMetric")
println(" # Aggregation method: $aggregationMethod")
println(" # Error function: $errorFunction")
println(" # Neighborhood size: [$knnMin:$knnMax]")
println(" # Neighborhood step: $knnStep")
println("")


testErrorsMean = []
testErrorsStdDev = []

for k in knnMin:knnStep:knnMax # foreach hyperparameter
    println("- Running for neighborhood size k = $k")

    testSetItemCount = 0
    kFoldErrors = []
    precisionSum = 0
    recallSum = 0
    fMeasureSum = 0
    perfectPredictionsSum = 0

    for kFoldIndex = 0:numberOfFolds-1

        println("\t- Iteration $(kFoldIndex + 1)/$numberOfFolds")

        # Training and test set splitting
        trainingDataFrame, testDataFrame = datasetSplit(ratingsDataFrame, testSetSize, kFoldIndex)
        
        # Building the URM
        trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)
        testURM = buildURM(testDataFrame, numberOfUsers, numberOfMovies)
        urmDensity = getDensityPercentage(trainingURM, trainingDataFrame)
        testSetItemCount = size(testDataFrame, 1)

        println("\t\t- URM density: $urmDensity%")

        # Compute targets
        targets, predictions = computePredictions(trainingURM, testDataFrame, testURM, aggregationMethod, k, similarityMetric)

        # Compute performance metrics on current fold
        mae = errorFunction(targets, predictions)
        _precision, recall = computePrecisionAndRecall(targets, predictions)
        fMeasure = (2 * _precision * recall) / (_precision + recall)
        perfectPredictions = computeNumberOfPerfectPredictions(targets, predictions)

        precisionSum += _precision
        recallSum += recall 
        fMeasureSum += fMeasure
        perfectPredictionsSum += perfectPredictions

        println("\t\t- Test error: $mae")

        push!(kFoldErrors, mae)

        GC.gc(true) # Explicit call to the garbage collector to make sure no memory is leaked
    end

    # Compute test error as the average of test errors on each folds
    avgValError = mean(kFoldErrors)
    stdDevError = std(kFoldErrors)
    precisionMean = precisionSum / numberOfFolds
    recallMean = recallSum / numberOfFolds
    fMeasureMean = fMeasureSum / numberOfFolds
    perfectPredictionsMean = perfectPredictionsSum / numberOfFolds
    perfectPredictionsPercentage = perfectPredictionsMean/testSetItemCount * 100

    println("\t# Avg test error: $(round(avgValError, digits=3))")
    println("\t# StdDev test error: $(round(stdDevError, digits=3))")
    println("\t# Precision mean: $(round(precisionMean, digits=3))")
    println("\t# Recall mean: $(round(recallMean, digits=3))")
    println("\t# f-measure mean: $(round(fMeasureMean, digits=3))")
    println("\t# Perfect predictions mean: $(round(perfectPredictionsMean, digits=3))/$testSetItemCount ($(round(perfectPredictionsPercentage, digits=3))%)")
    

    # Store the test error we just computed in testErrors
    push!(testErrorsMean, (k, avgValError))
    push!(testErrorsStdDev, (k, stdDevError))
end

println("✓ Operation completed")


CSV.write("$aggregationMethod error mean.csv", Tables.table(testErrorsMean), writeheaders=false)
CSV.write("$aggregationMethod error stddev.csv", Tables.table(testErrorsStdDev), writeheaders=false)


plotHistory(testErrorsMean, "Test error mean")
println("Press a key to continue...")
readline()
plotHistory(testErrorsStdDev, "Test error std dev")

# Performance evaluation
sort!(testErrorsMean, by = x -> x[2])
bestNeighborhoodSize = testErrorsMean[1][1]
println("\nModel selection...")
println(" ✓ Best neighborhood size k = $bestNeighborhoodSize")

