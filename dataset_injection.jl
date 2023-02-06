using CSV
using DataFrames

"""
Loads data from specified dataset

# Arguments
- `dataset::String`: dataset name (ml-latest, ml-latest-small, ml-100k, ml-1m, ml-10m, ml-20m, ml-25m)

# Returns
- `linksDataFrame::DataFrame`: links dataframe
- `moviesDataFrame::DataFrame`: movies dataframe
- `ratingsDataFrame::DataFrame`: ratings dataframe
- `tagsDataFrame::DataFrame`: tags dataframe
"""
function loadData(dataset)
    dataset_zip = dataset * ".zip"

    if !isfile(dataset_zip)
        url = "https://files.grouplens.org/datasets/movielens/" * dataset_zip
        download(url, dataset_zip)
        run(`unzip $dataset_zip`)
    end

    linksDataFrame = DataFrame(CSV.File(dataset * "/links.csv"))
    moviesDataFrame = DataFrame(CSV.File(dataset * "/movies.csv"))
    ratingsDataFrame = DataFrame(CSV.File(dataset * "/ratings.csv"))
    tagsDataFrame = DataFrame(CSV.File(dataset * "/tags.csv"))

    return linksDataFrame, moviesDataFrame, ratingsDataFrame, tagsDataFrame
end
