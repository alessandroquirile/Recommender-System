function computeModelError(trainingURM, targetDataFrame, targetURM, aggregationMethod, k, metric, errorFunction)
    testSetItemCount = size(targetDataFrame, 1)
    predictions = Vector{Union{Missing, Float64}}(undef, testSetItemCount)
    targets = targetDataFrame[:, :rating]
    normalize!(targets, getRatingRange())

    @threads for i = 1:testSetItemCount
        userId = targetDataFrame[i, :userId]
        movieId = targetDataFrame[i, :movieId]
        item = getMovieIndexById(movieId)

        user = targetURM[userId, :]
        predictions[i] = aggregationMethod(trainingURM, user, item, k, metric)
        
        # Explicit call to the Garbage Collecter to fix a memory leak
        GC.gc(false) # fullSweep = false means the GC will only clear "young" objects, speeding up the collection
    end
    
    error = errorFunction(targets, predictions)
    return error
end
