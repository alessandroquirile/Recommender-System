function computeModelError(trainingURM, targetDataFrame, targetURM, aggregationMethod, k, metric, errorFunction)
    testSetItemCount = size(targetDataFrame, 1)
    predictions = Vector{Union{Missing, Float64}}(undef, testSetItemCount)
    targets = targetDataFrame[:, :rating]
    targets = normalize(targets, getRatingRange())

    @threads for i = 1:testSetItemCount
        userId = targetDataFrame[i, :userId]
        movieId = targetDataFrame[i, :movieId]
        item = getMovieIndexById(movieId)

        user = targetURM[userId, :]
        predictions[i] = aggregationMethod(trainingURM, user, item, k, metric)
    end

    error = errorFunction(targets, predictions)
    return error
end
