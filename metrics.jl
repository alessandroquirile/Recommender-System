"""
Computes the Mean Squared Difference (MSD) based on provided inputs

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `msd`: mean squared differences between x and y

# Notes
- `x`, `y` in [0;1] implies `msd` in [0;1]
- High `msd` values implies high differences between `x` and `y`
"""
function msd(x, y)
    distance = d(x,y)
    skippedMissingDistance = collect(skipmissing(distance))
    if length(skippedMissingDistance) > 0
        msd = mean(skippedMissingDistance)
        return msd
    else
        return 1
    end
end

"""
Computes the Jaccard index based on provided inputs

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `jaccard`: Jaccard index
"""
function jaccard(x, y)
    distance = d(x,y)
    skippedMissingDistance = collect(skipmissing(distance))
    commonVotedItems = length(skippedMissingDistance)

    itemsXVoted = length(collect(skipmissing(x)))
    itemsYVoted = length(collect(skipmissing(y)))
    totalVotedItems = itemsXVoted + itemsYVoted - commonVotedItems

    if totalVotedItems > 0
        jaccard = commonVotedItems / totalVotedItems
        return jaccard
    end
end

"""
Computes a new similarity metric based on Jaccard and MSD indeces

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `newMetric`: new similarity metric
"""
function newMetric(x, y)
    newMetric = jaccard(x, y) * (1 - msd(x, y))
    return newMetric
end

"""
Computes the distance based on provided inputs

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `distance`: distance
"""
function d(x, y)
    distance = (x .- y).^2
    return distance
end