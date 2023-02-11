function buildURM(ratingsDataFrame, numberOfUsers, numberOfMovies, normalizeURM::Bool=true)
    URM = allocateMatrix(numberOfUsers, numberOfMovies)
    @threads for i=1:numberOfUsers
        userId = i
        userRatings = getUserRatings(ratingsDataFrame, userId)
        for j=1:numberOfMovies
            movieId = getMovieId(moviesDataFrame, j)
            URM[i,j] = getUserRatingByMovieId(userRatings, movieId)
        end
    end
    if normalizeURM
        URM = normalize(URM, getRatingRange())
    end
    return URM
end


function printInfo(urm)
    numberOfUsers, numberOfMovies = size(urm)
    println(" # URM shape is ($numberOfUsers, $numberOfMovies)")
end

function getDensity(urm, ratingsDataFrame)
    numberOfUsers, numberOfMovies = size(urm)
    numberOfRatings = size(ratingsDataFrame, 1)
    density = numberOfRatings / (numberOfMovies * numberOfUsers)

    return density
end

function getDensityPercentage(urm, ratingsDataFrame)
    density = getDensity(urm, ratingsDataFrame)
    percentage = density*100
    return round(percentage, digits=3)
end


function kNearestNeighbors(urm, userRatings, k, metric=newMetric)
    numberOfUsers = size(urm, 1)

    # Allocate the similarity matrix, having all the users as rows and 
    # storing the userId in the first column and the similarity metric in the second column
    userSimilaritiesMatrix = allocateMatrix(numberOfUsers, 2)
    userIdColumn = 1
    similarityColumn = 2


    # For each user in urm, append similarity to similarities
    for i=1:numberOfUsers
        similarity = metric(userRatings, urm[i, :])
        userSimilaritiesMatrix[i, userIdColumn] = i
        userSimilaritiesMatrix[i, similarityColumn] = similarity
    end

    # Sort similarities by the second column (similarity) descending order
    similarities = userSimilaritiesMatrix[sortperm( -userSimilaritiesMatrix[:, similarityColumn]), :]

    # Retrieve the first k indices (userIds)
    n = min(k, numberOfUsers) # if k > numberOfUsers, this prevents returning more items than there are in the matrix 
    return Int.(similarities[1:n, userIdColumn]) # returns the userIds (casted to Integers) ordered by similarity DESC
end

"""
Calculates a list of the knn ids w.r.t. user's ratings given to input item

# Arguments
- `URM`: the User Rating Matrix
- `userRatings`: user's ratings
- `k`: knn parameter
- `itemIndex`: item index in the URM
- `metric`: similarity metric

# Returns
- `knn`: list of knn ids
"""
function knnWhichRatedItem(URM, userRatings, k, itemIndex, metric=newMetric)
    # Get the neighborhood (userIds)
    knn = kNearestNeighbors(URM, userRatings, k, metric)
    knnWhichRatedItem = []

    # Filtro il vicinato selezionando solo gli utenti che hanno espresso almeno un rating (non missing) per item
    for n in eachindex(knn)
        neighborUserId = knn[n]
        if !ismissing(URM[neighborUserId, itemIndex])
            append!(knnWhichRatedItem, neighborUserId)
        end
    end

    # userIds
    return knnWhichRatedItem
end