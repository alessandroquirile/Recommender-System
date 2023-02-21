function getRatingAverages(dfGrouped::GroupedDataFrame, rounded::Bool=true)
    dfMean = combine(dfGrouped, :rating => x -> getMean(x, rounded))
    return dfMean
end

function getRatingModes(dfGrouped::GroupedDataFrame, rounded::Bool=true)
    dfModes = combine(dfGrouped, :rating => x -> getMode(x, rounded))
    return dfModes
end

function getRatingMedians(dfGrouped::GroupedDataFrame, rounded::Bool=true)
    dfMedians = combine(dfGrouped, :rating => x -> getMedian(x, rounded))
    return dfMedians
end
    

function getRatingStdDevs(dfGrouped::GroupedDataFrame, corrected::Bool, rounded::Bool=true)
    dfStdDev = combine(dfGrouped, :rating => x -> getStdDevOfNonMissingValues(x, corrected, rounded))
    return dfStdDev
end

function getMean(values, rounded::Bool=true)
    _mean = mean(values)
    if (rounded)
        _mean = round(_mean, digits=1)
    end
    return _mean
end

function getMode(values, rounded::Bool=true)
    _mode = mode(values)
    if (rounded)
        _mode = round(_mode, digits=1)
    end
    return _mode
end

function getMedian(values,  rounded::Bool=true)
    _median = median(values)
    if (rounded)
        _median = round(_median, digits=1)
    end
    return _median
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
    valueOfVotes = histogram(ratingsDataFrame.rating, title="Value of votes", label="", bar_width=0.4)

    dfGrouped = groupby(ratingsDataFrame, :movieId)

    # Arithmetic average histogram
    moviesRatingAverages = getRatingAverages(dfGrouped)
    arithmeticAverageHistogram = histogram(moviesRatingAverages.rating_function, title="Votes arithmetic average", label="")

    # Mode histogram
    #moviesRatingModes = getRatingModes(dfGrouped)
    #modeHistogram = histogram(moviesRatingModes.rating_function, title="Votes mode", label="")

    # Median histogram
    #moviesRatingMedian = getRatingMedians(dfGrouped)
    #medianHistogram = histogram(moviesRatingMedian.rating_function, title="Votes median", label="")

    # Standard deviation histogram
    moviesRatingStdDev = getRatingStdDevs(dfGrouped, true)
    stdDevHistogram = histogram(moviesRatingStdDev.rating_function, title="Votes standard deviation", label="")

    votesMode = mode(ratingsDataFrame.rating)
    votesMedian = median(ratingsDataFrame.rating)
    valueOfVotesSkewness = skewness(ratingsDataFrame.rating)

    # Plotting histograms
    showHistogram(valueOfVotes)
    showHistogram(arithmeticAverageHistogram)
    #showHistogram(modeHistogram)
    #showHistogram(medianHistogram)
    showHistogram(stdDevHistogram)

    println(" # Value of votes mode is $(round(votesMode, digits=3))")
    println(" # Value of votes median is $(round(votesMedian, digits=3))")
    println(" # Value of votes skewness is $(round(valueOfVotesSkewness, digits=3))")
end

# Plot validation errors
function plotHistory(validationErrors, title)
    xAxis = [x[1] for x in validationErrors]
    yAxis = [x[2] for x in validationErrors]

    yMin = round(minimum(yAxis), digits=3)
    yMax = round(maximum(yAxis), digits=3)
    yStep = round((yMax - yMin) / 16, digits=3)
    
    if (yStep == 0)
        yStep = 0.001
    end
    yTicks = yMin:yStep:yMax

    plot(xAxis, yAxis, show=true, title=title, xticks=xAxis, yticks=yTicks, xtickfontsize=5, ytickfontsize=5, label="")
    xlabel!("kNN size")
    ylabel!(title)
end
