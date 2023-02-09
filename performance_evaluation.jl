function computeModelError(trainingURM, targetDataFrame, targetURM, aggregationMethod, k, metric, errorFunction)
    testSetItemCount = size(targetDataFrame, 1)
    predictions = Vector{Union{Missing, Float64}}(undef, testSetItemCount)
    targets = targetDataFrame[:, :rating]
    normalize!(targets, getRatingRange())

    counter = Threads.Atomic{Int64}(0) # counter is an atomic variable of type Int64 initialized to 0
    @threads for i = 1:testSetItemCount
        Threads.atomic_add!(counter, 1) # increment counter
        progressPercentage = round(counter.value/testSetItemCount*100, digits=2)
        print("\rComputing prediction $(counter.value)/$testSetItemCount ($progressPercentage%)    ")

        userId = targetDataFrame[i, :userId]
        movieId = targetDataFrame[i, :movieId]
        item = getMovieIndexById(movieId)

        user = targetURM[userId, :]
        predictions[i] = aggregationMethod(trainingURM, user, item, k, metric)
        
        # Explicit call to the Garbage Collecter to fix a memory leak
        GC.gc(false) # fullSweep = false means the GC will only clear "young" objects, speeding up the collection
    end
    println("") # newline after progress report

    error = errorFunction(targets, predictions)
    return error
end
