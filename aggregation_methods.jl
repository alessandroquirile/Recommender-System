function averageAggregation(urm, userRatings, itemId, k, metric=newMetric)
    knnForItem = knnWhichRatedItem(urm, userRatings, k, itemId, metric)  # G_u,i

    if isempty(knnForItem)
        return missing
    end

    neighborsRatings = urm[knnForItem, itemId]  # r_n,i
    return mean(neighborsRatings)
end


function weightedSumAggregation(urm, userRatings, itemId, k, metric=newMetric)
    knnForItem = knnWhichRatedItem(urm, userRatings, k, itemId, metric)  # G_u,i

    if isempty(knnForItem)
        return missing
    end
        
    sum = 0.0
    similarities_sum = 0.0
    
    for neighborId in eachindex(knnForItem)
        neighborRatings = urm[knnForItem[neighborId], :]
        neighborRating = neighborRatings[itemId]  # r_n,i

        similarity = metric(userRatings, neighborRatings)
        similarities_sum = similarities_sum + similarity
        sum += similarity * neighborRating
    end
    
    if similarities_sum == 0
        return missing
    end
    
    mi = 1 / similarities_sum
    return sum * mi
end


function adjustedWeightedSumAggregation(urm, userRatings, itemId, k, metric=newMetric)
    knnForItem = knnWhichRatedItem(urm, userRatings, k, itemId, metric)  # G_u,i

    if isempty(knnForItem)
        return missing
    end

    sum = 0.0
    similarities_sum = 0.0

    for neighborId in eachindex(knnForItem)
        neighborRatings = urm[knnForItem[neighborId], :]
        neighborRating = neighborRatings[itemId]  # r_n,i
        neighborAvgRating = mean(collect(skipmissing(neighborRatings)))

        similarity = metric(userRatings, neighborRatings)
        similarities_sum = similarities_sum + similarity
        sum += similarity * (neighborRating - neighborAvgRating)
    end

    if similarities_sum == 0
        return missing
    end

    userAvgRating = mean(collect(skipmissing(userRatings)))
    mi = 1 / similarities_sum
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