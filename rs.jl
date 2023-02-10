function buildURM(ratingsDataFrame, numberOfUsers, numberOfMovies, normalizeURM::Bool=true)
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
        normalize!(URM, getRatingRange())
    end
    return URM
end


function printInfo(urm)
    numberOfUsers, numberOfMovies = size(urm)
    println(" # URM shape is ($numberOfUsers, $numberOfMovies)")
end

function getUrmDensity(urm, ratingsDataFrame)
    numberOfUsers, numberOfMovies = size(urm)
    numberOfRatings = size(ratingsDataFrame, 1)
    density = numberOfRatings / (numberOfMovies * numberOfUsers)

    return density
end

function getUrmDensityPercentage(urm, ratingsDataFrame)
    density = getUrmDensity(urm, ratingsDataFrame)
    percentage = density*100
    return round(percentage, digits=3)
end


function kNearestNeighbors(trainingURM, user, k, metric=newMetric)
    numberOfUsers = size(trainingURM, 1)
    similarities = Matrix{Union{Missing, Float32}}(missing, numberOfUsers, 2)  # (numberOfUsers, 2) shape

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

"""
Calculates a list of the knn ids w.r.t. user's ratings given to input item

# Arguments
- `trainingURM`: the training URM
- `user`: user's ratings
- `k`: knn parameter
- `item`: item id
- `metric`: similarity metric

# Returns
- `knn`: list of knn ids
"""
function knnWhichRatedItem(trainingURM, user, k, item, metric=newMetric)
    # Ottengo il vicinato (userIds)
    knn = kNearestNeighbors(trainingURM, user, k, metric)
    knnWhichRatedItem = []

    # Filtro il vicinato selezionando solo gli utenti che hanno espresso almeno un rating (non missing) per item
    for n in eachindex(knn)
        if !ismissing(trainingURM[knn[n], item])
            append!(knnWhichRatedItem, knn[n])
        end
    end

    # userIds
    return knnWhichRatedItem
end