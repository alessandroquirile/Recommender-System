using Statistics
using Plots
using Base.Threads
include("dataset_injection.jl")
include("rs.jl")
include("dataset_analysis.jl")
include("metrics.jl")

# Data injection
moviesDataFrame, ratingsDataFrame = loadDataSlim("ml-latest-small")

numberOfUsers = length(unique(ratingsDataFrame[:, 1])) # Number of unique "userId" values in ratingsDataFrame
numberOfMovies = size(moviesDataFrame, 1) # Number of rows in moviesDataFrame

# Building the URM
println("# Building the User Rating Matrix (URM)...")
urm = buildURM(numberOfUsers, numberOfMovies)
printInfo(urm)
printDensity(urm, ratingsDataFrame)

# Value of votes histogram
valueOfVotes = histogram(ratingsDataFrame.rating, title="Value of votes")

# Arithmetic average histogram
moviesRatingAverage = getAverage(ratingsDataFrame, :movieId)
arithmeticAverageHistogram = histogram(moviesRatingAverage.rating_function, title="Arithmetic average") # TODO: change x axis step size

# Standard deviation histogram
moviesRatingStdDev = getStdDev(ratingsDataFrame, :movieId, true)
stdDevHistogram = histogram(moviesRatingStdDev.rating_function, title="Standard deviation") # TODO: change x axis step size

# Plotting histograms
showHistogram(valueOfVotes)
showHistogram(arithmeticAverageHistogram)
showHistogram(stdDevHistogram)