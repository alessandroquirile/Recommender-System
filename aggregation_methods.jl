function averageAggregation(urm, user, item, k, metric=newMetric)
    knnWhichRatedItem = getUserNeighborsWhichRatedItem(urm, user, k, item, metric)
    if length(knnWhichRatedItem) != 0 
        itemRatingsGivenByNeighbors = urm[knnWhichRatedItem, item]
        return mean(itemRatingsGivenByNeighbors)
    end

    return missing
end


function weightedSumAggregation(urm, user, item, k, metric=newMetric)
    sum = 0.0
    knnWhichRatedItem = getUserNeighborsWhichRatedItem(urm, user, k, item, metric)
    if length(knnWhichRatedItem) != 0 
        for n in eachindex(knnWhichRatedItem)
            currentNeighbor = urm[knnWhichRatedItem[n], :]
            itemRatingGivenByCurrentNeighbor = currentNeighbor[item]

            similarity = metric(user, currentNeighbor)
            sum += similarity * itemRatingGivenByCurrentNeighbor
        end

        mi = normalizingFactor(urm, knnWhichRatedItem, user, metric)
        return sum * mi
    end

    return missing
end


function adjustedWeightedSumAggregation(urm, user, item, k, metric=newMetric)
    sum = 0.0
    knnWhichRatedItem = getUserNeighborsWhichRatedItem(urm, user, k, item, metric)
    if length(knnWhichRatedItem) != 0 
        for n in eachindex(knnWhichRatedItem)
            currentNeighbor = urm[knnWhichRatedItem[n], :]
            itemRatingGivenByCurrentNeighbor = currentNeighbor[item]
            meanRatingsOfCurrentNeighbor = mean(collect(skipmissing(currentNeighbor)))

            similarity = metric(user, currentNeighbor)
            sum += similarity * (itemRatingGivenByCurrentNeighbor - meanRatingsOfCurrentNeighbor)
        end

        meanRatingsOfGivenUser = mean(collect(skipmissing(user)))
        mi = normalizingFactor(urm, knnWhichRatedItem, user, metric)
        return meanRatingsOfGivenUser + sum * mi
    end

    return missing
end


function normalizingFactor(trainingURM, knnWhichRatedItem, user, metric=newMetric)
    sum = 0.0
    if length(knnWhichRatedItem) != 0 
        for n in eachindex(knnWhichRatedItem)
            user_n = trainingURM[knnWhichRatedItem[n], :]
            similarity = metric(user, user_n)
            sum += similarity
        end
        return 1.0 / sum
    end
end
