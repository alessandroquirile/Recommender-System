function buildURM()
    # Initializes a Matrix of types {Missing or Float32} initialized to the value missing and has shape (numberOfUsers, numberOfMovies)
    URM = Matrix{Union{Missing, Float32}}(missing, numberOfUsers, numberOfMovies)

    for i=1:numberOfUsers
        userId = i
        userRatings = ratingsDataFrame[ratingsDataFrame.userId .== userId, :] # select all the ratings given by the current user

        for j=1:numberOfMovies
            movieId = moviesDataFrame[j, 1]

            # find the rating given to the current movie 
            rating = userRatings[userRatings.movieId .== movieId, 3] # rating is empty is no rating is found, a singleton if the rating is found

            if length(rating) > 0 #if the user has rated the movie
                URM[i,j] = rating[1]
            end
        end
    end

    return URM
end


function getMovieIndexById(movieId)
    for i=1:numberOfMovies
        if moviesDataFrame[i,1] == movieId
            return i
        end
    end
    throw(KeyError(movieId))
end