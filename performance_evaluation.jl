function computeModelError(trainingURM, targetDataFrame, targetURM, aggregationMethod, k, metric)
    testSetItemCount = size(targetDataFrame, 1)

    # Extract and normalize target ratings
    targets = targetDataFrame[:, :rating]
    targets = normalize(targets, getRatingRange())

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
    
    return targets, predictions
end


function computePrecisionAndRecall(targets, predictions)
    theta = 0.75 # Every vote >= 0.75 is considered positive, else is considered negative

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i = eachindex(targets)
        t = targets[i]
        y = predictions[i]
        if (ismissing(predictions[i]))
            continue
        end

        if t >= theta # if target is positive
            if y >= theta # if prediciton is positive
                TP = TP + 1
            else #prediction is negative
                FN = FN + 1
            end
        else # target is negative
            if y >= theta # if prediciton is positive
                FP = FP + 1
            else #prediction is negative
                TN = TN + 1
            end
        end
    end

    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)

    return precision, recall
end

function computeNumberOfPerfectPredictions(targets, predictions)
    counter = 0
    for i=eachindex(targets)
        if !ismissing(predictions[i]) && abs(targets[i] - predictions[i]) < 0.125
            counter = counter + 1
        end
    end
    return counter
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
