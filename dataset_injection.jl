using CSV
using DataFrames


"""
Loads data from specified dataset

# Arguments
- `dataset::String`: dataset name (ml-latest, ml-latest-small, ml-100k, ml-1m, ml-10m, ml-20m, ml-25m)

# Returns
- `moviesDataFrame::DataFrame`: movies dataframe
- `ratingsDataFrame::DataFrame`: ratings dataframe
"""
function loadData(dataset)
    downloadDataset(dataset)

    println("Loading $dataset dataset...")

    if isfile(dataset * "/movies.csv")
        moviesDataFrame = DataFrame(CSV.File(dataset * "/movies.csv"))
        ratingsDataFrame = DataFrame(CSV.File(dataset * "/ratings.csv"))
    else
        moviesHeader = Vector{String}("movieId", "imdbId", "tmdbId")
        ratingsHeader = Vector{String}("userId", "movieId", "rating", "timestamp")

        moviesDataFrame = DataFrame(CSV.File(dataset * "/movies.dat", delim = "::", header = moviesHeader))
        ratingsDataFrame = DataFrame(CSV.File(dataset * "/ratings.dat", delim = "::", header = ratingsHeader))
    end

    println("✓ Dataset loaded\n")
    return moviesDataFrame, ratingsDataFrame
end

function downloadDataset(dataset)
    dataset_zip = dataset * ".zip"

    if !isfile(dataset_zip)
        println("Downloading " * dataset_zip)
        url = "https://files.grouplens.org/datasets/movielens/" * dataset_zip
        download(url, dataset_zip)
        run(`unzip $dataset_zip`)
    end
end

function getRatingRange()
    return 0.5:5
end