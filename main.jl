using Statistics
using Plots
using Base.Threads
include("dataset_injection.jl")
include("rs.jl")
include("dataset_analysis.jl")

linksDataFrame, moviesDataFrame, ratingsDataFrame, tagsDataFrame = loadData("ml-latest-small")

numberOfMovies = size(moviesDataFrame, 1) # number of rows in moviesDataFrame
numberOfRatings = size(ratingsDataFrame, 1) # number of rows in ratingsDataFrame
numberOfUsers = length(unique(ratingsDataFrame[:,1])) # number of unique "userId" values in ratingsDataFrame

println("Building the User Rating Matrix...")
URM = buildURM()

density = numberOfRatings / (numberOfMovies * numberOfUsers)
println("Loaded $numberOfMovies movies and $numberOfUsers users")
println("Found $numberOfRatings ratings, URM matrix density is $(round(density*100, digits=3))%")


#Print the Value of votes histogram
valueOfVotes = histogram(ratingsDataFrame.rating, title="Value of votes")
#plot(valueOfVotes, show=true)


#Arithmetic average
moviesRatingAverage = getRoundedAverage(ratingsDataFrame, :movieId)
arithmeticAverageHistogram = histogram(moviesRatingAverage.rating_function, title="Arithmetic average") # TODO: change x axis step size


moviesRatingStdDev = getRoundedStdDev(ratingsDataFrame, :movieId, true)
stdDevHistogram = histogram(moviesRatingStdDev.rating_function, title="Standard deviation") # TODO: change x axis step size

showHistogram(valueOfVotes)
showHistogram(arithmeticAverageHistogram)
showHistogram(stdDevHistogram)
