function getRoundedAverage(dataFrame::DataFrame, groupByCols)
    dfGrouped = groupby(dataFrame, groupByCols)
    dfMean = combine(dfGrouped, :rating => x -> round(mean(x), digits=1))
    return dfMean
end

function getRoundedStdDev(dataFrame::DataFrame, groupByCols, corrected::Bool)
    dfGrouped = groupby(dataFrame, groupByCols)
    dfStdDev = combine(dfGrouped, :rating => x -> round(std(collect(skipmissing(x)), corrected=corrected), digits=1))
    return dfStdDev
end

function showHistogram(histogram)
    plot(histogram, show=true)
    println("Press a key to continue...")
    readline()
end