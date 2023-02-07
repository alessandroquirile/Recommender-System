function averageAggregation(trainingURM, user, k, item)
    knnWhichRatedItem =  getUserNeighborsWhichRatedItem(trainingURM, user, k, item)
    if length(knnWhichRatedItem) != 0 
        ratings = trainingURM[knnWhichRatedItem, item]
        return mean(ratings)
    end
end


function weightedSumAggregation(trainingURM, user, item, k, metric=newMetric)
    sum = 0
    knnWhichRatedItem =  getUserNeighborsWhichRatedItem(trainingURM, user, k, item)
    if length(knnWhichRatedItem) != 0 
        for i in eachindex(knnWhichRatedItem)
            user_i = trainingURM[knnWhichRatedItem[i], :]
            similarity = metric(user, user_i)
            sum += similarity * trainingURM[knnWhichRatedItem[i], item]
        end
        return sum * normalizingFactor(trainingURM, knnWhichRatedItem, user, metric)
    end
end


function adjustedWeightedSumAggregation(trainingURM, user, item, k, metric=newMetric)
    sum = 0
    knnWhichRatedItem =  getUserNeighborsWhichRatedItem(trainingURM, user, k, item)
    if length(knnWhichRatedItem) != 0 
        for i in eachindex(knnWhichRatedItem)
            user_i = trainingURM[knnWhichRatedItem[i], :]
            similarity = metric(user, user_i)
            sum += similarity * (trainingURM[knnWhichRatedItem[i], item] - mean(collect(skipmissing(user_i))))
        end
        return mean(collect(skipmissing(user))) + sum * normalizingFactor(trainingURM, knnWhichRatedItem, user, metric)
    end
end


function normalizingFactor(trainingURM, knnWhichRatedItem, user, metric=newMetric)
    sum = 0
    if length(knnWhichRatedItem) != 0 
        for i in eachindex(knnWhichRatedItem)
            user_i = trainingURM[knnWhichRatedItem[i], :]
            similarity = metric(user, user_i)
            sum += similarity
        end
        return 1 / sum
    end
end