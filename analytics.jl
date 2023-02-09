function getAverage(dataFrame::DataFrame, groupByCols, rounded::Bool=true)
    dfGrouped = groupby(dataFrame, groupByCols)
    dfMean = combine(dfGrouped, :rating => x -> getMean(x, rounded))

    return dfMean
end


function getStdDev(dataFrame::DataFrame, groupByCols, corrected::Bool, rounded::Bool=true)
    dfGrouped = groupby(dataFrame, groupByCols)
    dfStdDev = combine(dfGrouped, :rating => x -> getStdDevOfNonMissingValues(x, corrected, rounded))

    return dfStdDev
end

function getMean(values,  rounded::Bool=true)
    mean = mean(values)
    if (rounded)
        mean = round(mean, digits=1)
    end
    return mean
end

function getStdDevOfNonMissingValues(values, corrected::Bool, rounded::Bool=true)
    nonMissingValues = collect(skipmissing(values))
    stdDev = std(nonMissingValues, corrected=corrected)

    if (rounded)
        stdDev = round(stdDev, digits=1)
    end

    return stdDev
end


function showHistogram(histogram)
    plot(histogram, show=true)
    println("Press a key to continue...")
    readline()
end


function printStatistics()
    # Value of votes histogram
    valueOfVotes = histogram(ratingsDataFrame.rating, title="Value of votes")

    # Arithmetic average histogram
    moviesRatingAverage = getAverage(ratingsDataFrame, :movieId)
    arithmeticAverageHistogram = histogram(moviesRatingAverage.rating_function, title="Arithmetic average") # TODO: change x axis step size

    # Standard deviation histogram
    moviesRatingStdDev = getStdDev(ratingsDataFrame, :movieId, true)
    stdDevHistogram = histogram(moviesRatingStdDev.rating_function, title="Standard deviation") # TODO: change x axis step size

    # Plotting histograms
    showHistogram(valueOfVotes)
    showHistogram(arithmeticAverageHistogram)
    showHistogram(stdDevHistogram)
end

# Plot validation errors
function plotValidationHistory(validationErrors)
    xAxis = [x[1] for x in validationErrors]
    yAxis = [x[2] for x in validationErrors]
    plot(xAxis, yAxis, title="Validation errors (MAE)", show=true)
    xlabel!("kNN size")
    ylabel!("Validation error")
end
