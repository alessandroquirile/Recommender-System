function getMoviesRatingAverage(moviesDataFrame::DataFrame, ratingsDataFrame::DataFrame)
    movieRatingJoin = innerjoin(moviesDataFrame, ratingsDataFrame, on = :movieId)
    dfGrouped = groupby(movieRatingJoin, :movieId)
    dfMean = combine(dfGrouped, :rating => mean)
    dfMean[!, :rating_mean] = round.(dfMean[!, :rating_mean], digits=1)
    return dfMean
end
