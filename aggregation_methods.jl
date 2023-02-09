function averageAggregation(urm, userRatings, itemId, k, metric=newMetric)
    knnForItem = knnWhichRatedItem(urm, userRatings, k, itemId, metric)  # G_u,i

    if isempty(knnForItem)
        return missing
    end

    neighborsRating = urm[knnForItem, itemId]  # r_n,i
    return mean(neighborsRating)
end


function weightedSumAggregation(urm, userRatings, itemId, k, metric=newMetric)
    knnForItem = knnWhichRatedItem(urm, userRatings, k, itemId, metric)  # G_u,i

    if isempty(knnForItem)
        return missing
    end
        
    sum = 0.0
    for neighborId in eachindex(knnForItem)
        neighborRatings = urm[knnForItem[neighborId], :]
        neighborRating = neighborRatings[itemId]  # r_n,i

        similarity = metric(userRatings, neighborRatings)
        sum += similarity * neighborRating
    end
    
    mi = normalizingFactor(urm, knnForItem, userRatings, metric)
    return sum * mi
end


function adjustedWeightedSumAggregation(urm, userRatings, itemId, k, metric=newMetric)
    knnForItem = knnWhichRatedItem(urm, userRatings, k, itemId, metric)  # G_u,i

    if isempty(knnForItem)
        return missing
    end

    sum = 0.0
    for neighborId in eachindex(knnForItem)
        neighborRatings = urm[knnForItem[neighborId], :]
        neighborRating = neighborRatings[itemId]  # r_n,i
        neighborAvgRating = mean(collect(skipmissing(neighborRatings)))

        similarity = metric(userRatings, neighborRatings)
        sum += similarity * (neighborRating - neighborAvgRating)
    end

    userAvgRating = mean(collect(skipmissing(userRatings)))
    mi = normalizingFactor(urm, knnForItem, userRatings, metric)
    return userAvgRating + sum * mi
end


function normalizingFactor(trainingURM, knnForItem, userRatings, metric=newMetric)
    sum = 0.0
    for neighborId in eachindex(knnForItem)
        neighborRatings = trainingURM[knnForItem[neighborId], :]
        similarity = metric(userRatings, neighborRatings)
        sum += similarity
    end

    if (sum == 0.0)
        return missing
    else
        return 1.0 / sum
    end
end