function computeModelError(trainingURM, targetDataFrame, targetURM, aggregationMethod, k, metric, errorFunction)
    testSetItemCount = size(targetDataFrame, 1)

    # Extract and normalize target ratings
    targets = targetDataFrame[:, :rating]
    normalize!(targets, getRatingRange())

    # Allocate predictions vector
    predictions = Vector{Union{Missing, Float64}}(undef, testSetItemCount)
    
    # We keep track of how long the predictions took. The variable is atomic because the following for loop is multithreaded
    totalTime = Threads.Atomic{Float64}(0) 

    # We keep track of how many predictions we computed so far to give a progress report
    counter = Threads.Atomic{Int64}(0) # counter is an atomic variable of type Int64 initialized to 0

    # for each test set item
    @threads for i = 1:testSetItemCount

        # Print progress
        printErrorProgress(counter, testSetItemCount)

        # Extract relevant information of the current prediction
        userId = targetDataFrame[i, :userId]
        movieId = targetDataFrame[i, :movieId]
        itemIndex = getMovieIndexById(movieId)

        user = targetURM[userId, :]

        timeBefore = time() # time, in seconds, before the prediction started
        predictions[i] = aggregationMethod(trainingURM, user, itemIndex, k, metric) # compute prediction

        # We now calculate how long the prediciton took and add it to the totalTime atomic variable
        elapsedTime = time() - timeBefore
        Threads.atomic_add!(totalTime, elapsedTime) 

        # Explicit call to the Garbage Collecter to fix a memory leak
        GC.gc(false) # fullSweep = false means the GC will only clear "young" objects, speeding up the collection
    end

    println("") # newline after progress report
    printTimeReport(totalTime, testSetItemCount)
    
    error = errorFunction(targets, predictions)
    return error
end

function printErrorProgress(counter, testSetItemCount)
    Threads.atomic_add!(counter, 1) # increment counter
    progressPercentage = round(counter.value/testSetItemCount*100, digits=2)
    print("\r\t\t- Computing prediction $(counter.value)/$testSetItemCount ($progressPercentage%)    ")
end

function printTimeReport(totalTime, testSetItemCount)
    roundedTotalTime = round(totalTime.value, digits=3)
    timePerPrediction = round(totalTime.value / testSetItemCount, digits=3)
    println("\t\t- Computed $testSetItemCount predictions in $(roundedTotalTime)s: $timePerPrediction s/prediction")
end
