"""
Computes the Mean Squared Difference (MSD) based on provided inputs
`x` and `y` must have the same size

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `msd`: mean squared differences between x and y

# Notes
- `x`, `y` in [0;1] implies `msd` in [0;1]
- High `msd` values implies high differences between `x` and `y`
"""
function meanSquaredDifference(x, y)
    return mean(skipmissing(squaredDifference(x, y)))
end

"""
Computes the Jaccard index based on provided inputs
`x` and `y` must have the same size

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `jaccard`: Jaccard index
"""
function jaccard(x, y)
    d = squaredDifference(x, y)
    numberOfNonMissingDistances = length(collect(skipmissing(d)))
    numberOfNonMissingValuesInX = length(collect(skipmissing(x)))
    numberOfNonMissingValuesInY = length(collect(skipmissing(y)))

    return numberOfNonMissingDistances / (numberOfNonMissingValuesInX + numberOfNonMissingValuesInY - numberOfNonMissingDistances)
end

"""
Computes a new similarity metric based on Jaccard and MSD indeces

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `newMetric`: a new similarity metric based on Jaccard and MSD
"""
function newMetric(x, y)
    return jaccard(x, y) * (1 - meanSquaredDifference(x, y))
end

"""
Computes the squared difference based on provided inputs

# Arguments
- `x`: first user's ratings
- `y`: second user's ratings

# Returns
- `distance`: distance
"""
function squaredDifference(x, y)
    return (x .- y).^2
end


"""
Computes the Mean Absolute Error (MAE) based on provided inputs

# Arguments
- `target`: target vector
- `prediction`: prediction vector

# Returns
- `mae`: mean absolute error
"""
function meanAbsoluteError(target, prediction)
    difference = skipmissing(target - prediction)
    absoluteDifference = broadcast(abs, difference)
    return mean(absoluteDifference)
end