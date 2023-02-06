using Statistics
using Plots
include("dataset_injection.jl")
include("rs.jl")

linksDataFrame, moviesDataFrame, ratingsDataFrame, tagsDataFrame = loadData("ml-latest-small")

numberOfMovies = size(moviesDataFrame, 1) # number of rows in moviesDataFrame
numberOfRatings = size(ratingsDataFrame, 1) # number of rows in ratingsDataFrame
numberOfUsers = length(unique(ratingsDataFrame[:,1])) # number of unique "userId" values in ratingsDataFrame

URM = buildURM()

density = numberOfRatings / (numberOfMovies * numberOfUsers)
println("Loaded $numberOfMovies movies and $numberOfUsers users")
println("Found $numberOfRatings ratings, URM matrix density is $(density*100)%")