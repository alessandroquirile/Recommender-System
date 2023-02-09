function allocateMatrix(rows, columns)
    return Matrix{Union{Missing, Float32}}(missing, rows, columns)
end


function getMovieIndexById(movieId)
    for i=1:numberOfMovies
        if moviesDataFrame[i,1] == movieId
            return i
        end
    end
    throw(KeyError(movieId))
end


function normalize!(itr, range)
    min = minimum(range)
    max = maximum(range)
    itr = (itr .- min) / (max - min)
end


function getUserRatings(ratingsDataFrame, userId)
    return ratingsDataFrame[ratingsDataFrame.userId .== userId, :]
end


function getMovieId(moviesDataFrame, currentMovie)
    idColumn = 1
    return moviesDataFrame[currentMovie, idColumn]
end


function getUserRatingByMovieId(userRatings, movieId)
    ratingColumn = 3
    return userRatings[userRatings.movieId .== movieId, ratingColumn]
end


function isEmpty(rating)
    return length(rating) == 0
end