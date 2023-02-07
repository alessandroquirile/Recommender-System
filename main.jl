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


printStatistics()


testSetSize = 0.10
for kFoldIndex = 0:9
    println("Running fold number $kFoldIndex")

    # Splitting in training and test set
    trainingDataFrame, testDataFrame = kFoldSplit(ratingsDataFrame, numberOfUsers, numberOfMovies, testSetSize, kFoldIndex)
    # Building the URM
    trainingURM = buildURM(trainingDataFrame, numberOfUsers, numberOfMovies)

    printInfo(trainingURM)
    printDensity(trainingURM, trainingDataFrame)
end




