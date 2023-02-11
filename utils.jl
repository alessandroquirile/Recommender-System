function allocateMatrix(rows, columns)
    return Matrix{Union{Missing, Float32}}(missing, rows, columns)
end

"""
Converts a item index (as the column number in the URM) to its movie ID in the MovieLens database
"""
function getMovieIndexById(movieId)
    for i=1:numberOfMovies
        if moviesDataFrame[i,1] == movieId
            return i
        end
    end
    throw(KeyError(movieId))
end


function normalize(itr, range)
    min = minimum(range)
    max = maximum(range)
    return (itr .- min) / (max - min)
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
    ratingSingleton = userRatings[userRatings.movieId .== movieId, ratingColumn]

    if (isempty(ratingSingleton))
        return missing
    else
        return ratingSingleton[1]
    end
end
