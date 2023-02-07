function buildURM(ratingsDataFrame, numberOfUsers, numberOfMovies)
    println("# Building the User Rating Matrix (URM)...")
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


function printInfo(urm)
    numberOfUsers, numberOfMovies = size(urm)
    println("Loaded $numberOfUsers users and $numberOfMovies movies")
    println("URM shape is ($numberOfUsers, $numberOfMovies)")
end


function printDensity(urm, ratingsDataFrame)
    numberOfUsers, numberOfMovies = size(urm)
    numberOfRatings = size(ratingsDataFrame, 1)
    density = numberOfRatings / (numberOfMovies * numberOfUsers)
    println("Found $numberOfRatings ratings")
    println("URM density is $(round(density*100, digits=3))%")
    println("\n")
end


function kNearestNeighbors(trainingURM, user, k, metric=newMetric)
    numberOfUsers = size(trainingURM, 1)
    similarities = Matrix{Union{Missing, Float32}}(missing, numberOfUsers, 2)

    for i=1:numberOfUsers
        similarity = metric(user, trainingURM[i,:])
        similarities[i, 1] = i
        similarities[i, 2] = similarity
    end

    similarities = similarities[sortperm(similarities[:,2]), :]

    n = min(k, numberOfUsers)
    return similarities[1:n, 2]
end