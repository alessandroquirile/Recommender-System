function buildURM(ratingsDataFrame, numberOfUsers, numberOfMovies, normalizeURM::Bool=true)
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
    if normalizeURM
        URM = normalize(URM, getRatingRange())
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
    # (numberOfUsers, 2) shape
    similarities = Matrix{Union{Missing, Float32}}(missing, numberOfUsers, 2)

    # For each user in trainingURM, append similarity to similarities
    userIdColumn = 1
    similarityColumn = 2
    for i=1:numberOfUsers
        similarity = metric(user, trainingURM[i, :])
        similarities[i, userIdColumn] = i
        similarities[i, similarityColumn] = similarity
    end

    # Sort similarities by the second column (similarity) descending order
    similarities = similarities[sortperm(-similarities[:, similarityColumn]), :]

    # Retrieve the first k indices (userIds)
    n = min(k, numberOfUsers)
    return Int.(similarities[1:n, userIdColumn])
end


function getUserNeighborsWhichRatedItem(trainingURM, user, k, item, metric=newMetric)
    # Ottengo il vicinato (userIds)
    knn = kNearestNeighbors(trainingURM, user, k, metric)
    knnWhichRatedItem = []

    # Filtro il vicinato selezionado soltando gli utenti che hanno espresso almeno un rating (non missing) per item
    for i in eachindex(knn)
        if !ismissing(trainingURM[knn[i], item])
            append!(knnWhichRatedItem, knn[i])
        end
    end

    # userIds
    return knnWhichRatedItem
end