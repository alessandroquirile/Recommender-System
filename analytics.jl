function getAverage(dataFrame::DataFrame, groupByCols, rounded::Bool=true)
    dfGrouped = groupby(dataFrame, groupByCols)
    dfMean = combine(dfGrouped, :rating => x -> mean(x))
    if rounded
        dfMean = combine(dfGrouped, :rating => x -> round(mean(x), digits=1))
    end
    return dfMean
end


function getStdDev(dataFrame::DataFrame, groupByCols, corrected::Bool, rounded::Bool=true)
    dfGrouped = groupby(dataFrame, groupByCols)
    dfStdDev = combine(dfGrouped, :rating => x -> std(collect(skipmissing(x)), corrected=corrected))
    if rounded
        dfStdDev = combine(dfGrouped, :rating => x -> round(std(collect(skipmissing(x)), corrected=corrected), digits=1))
    end
    return dfStdDev
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
    plot(xAxis, yAxis, title="Validation errors (MAE)")
    xlabel!("kNN size")
    ylabel!("Validation error")
end
