function buildURM()
    URM = allocateMatrix(numberOfUsers, numberOfMovies)
    @threads for i=1:numberOfUsers
        userId = i
        userRatings = getUserRatings(ratingsDataFrame, userId)
        for j=1:numberOfMovies
            movieId = getMovieId(moviesDataFrame, j)
            rating = getUserRatingByMovieId(userRatings, movieId)
            if !isEmpty(rating)
                URM[i,j] = rating[1]
            end
        end
    end
    return URM
end


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


function normalize(itr, min=1, max=5)
    normalized = (itr .- min) / (max - min)
    return normalized
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