function computeModelError(aggregationMethod, k, metric, errorFunction)
    testSetItemCount = size(testDataFrame, 1)
    predictions = Vector{Union{Missing, Float64}}(undef, testSetItemCount)
    targets = testDataFrame[:, :rating]
    targets = normalize(targets, getRatingRange())

    @threads for i = 1:testSetItemCount
        userId = testDataFrame[i, :userId]
        movieId = testDataFrame[i, :movieId]
        item = getMovieIndexById(movieId)

        user = testURM[userId, :]
        predictions[i] = aggregationMethod(trainingURM, user, item, k, metric)
    end

    error = errorFunction(targets, predictions)
    return error
end

function evaluatePerformance()
    k = 10
    metric = newMetric
    aggregationMethod = averageAggregation
    errorFunction = meanAbsoluteError
    
    error = computeModelError(aggregationMethod, k, metric, errorFunction)
    println("MAE for averageAggregation, newMetric, k=$k: $error")
end