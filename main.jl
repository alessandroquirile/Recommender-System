using Statistics
using Plots
using Base.Threads
include("dataset_injection.jl")
include("rs.jl")
include("dataset_analysis.jl")
include("metrics.jl")

# Data injection
linksDataFrame, moviesDataFrame, ratingsDataFrame, tagsDataFrame = loadData("ml-latest-small")

numberOfMovies = size(moviesDataFrame, 1) # Number of rows in moviesDataFrame
numberOfRatings = size(ratingsDataFrame, 1) # Number of rows in ratingsDataFrame
numberOfUsers = length(unique(ratingsDataFrame[:,1])) # Number of unique "userId" values in ratingsDataFrame

# Building the URM
println("Building the User Rating Matrix (URM)...")
URM = buildURM()
density = numberOfRatings / (numberOfMovies * numberOfUsers)
println("Loaded $numberOfMovies movies and $numberOfUsers users")
println("Found $numberOfRatings ratings, URM matrix density is $(round(density*100, digits=3))%")

# Value of votes histogram
valueOfVotes = histogram(ratingsDataFrame.rating, title="Value of votes")

# Arithmetic average histogram
moviesRatingAverage = getRoundedAverage(ratingsDataFrame, :movieId)
arithmeticAverageHistogram = histogram(moviesRatingAverage.rating_function, title="Arithmetic average") # TODO: change x axis step size

# Standard deviation histogram
moviesRatingStdDev = getRoundedStdDev(ratingsDataFrame, :movieId, true)
stdDevHistogram = histogram(moviesRatingStdDev.rating_function, title="Standard deviation") # TODO: change x axis step size

# Plotting histograms
showHistogram(valueOfVotes)
showHistogram(arithmeticAverageHistogram)
showHistogram(stdDevHistogram)