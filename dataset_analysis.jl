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