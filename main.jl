using Statistics
using Plots
using Base.Threads
include("dataset_injection.jl")
include("rs.jl")
include("dataset_analysis.jl")
include("metrics.jl")
include("dataset_split.jl")

# Data injection
moviesDataFrame, ratingsDataFrame = loadDataSlim("ml-latest-small")

numberOfUsers = length(unique(ratingsDataFrame[:, 1])) # Number of unique "userId" values in ratingsDataFrame
numberOfMovies = size(moviesDataFrame, 1) # Number of rows in moviesDataFrame

# Print data set statistics
printStatistics()

# Training and test set splitting
testSetSize = 0.10
trainingAndValidationDataFrame, testDataFrame = kFoldSplit(ratingsDataFrame, numberOfUsers, numberOfMovies, testSetSize, 0)

for kFoldIndex = 0:9
    println("Running fold number $kFoldIndex")

    # Training and validation set splitting
    validationSetSize = 0.10
    trainingDataFrame, validationDataFrame = kFoldSplit(trainingAndValidationDataFrame, numberOfUsers, numberOfMovies, validationSetSize, kFoldIndex)

    # Building the URM
    trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)

    # Printing info
    printInfo(trainingURM)
    printDensity(trainingURM, trainingDataFrame)
end