using CSV
using DataFrames

"""
Loads data from specified dataset

# Arguments
- `dataset::String`: dataset name (ml-latest, ml-latest-small, ml-100k, ml-1m, ml-10m, ml-20m, ml-25m)

# Returns
- `links::DataFrame`: links dataframe
- `movies::DataFrame`: movies dataframe
- `ratings::DataFrame`: ratings dataframe
- `tags::DataFrame`: tags dataframe
"""
function loadData(dataset)
    dataset_zip = dataset * ".zip"

    if !isfile(dataset_zip)
        url = "https://files.grouplens.org/datasets/movielens/" * dataset_zip
        download(url, dataset_zip)
        run(`unzip $dataset_zip`)
    end

    links = DataFrame(CSV.File(dataset * "/links.csv"))
    movies = DataFrame(CSV.File(dataset * "/movies.csv"))
    ratings = DataFrame(CSV.File(dataset * "/ratings.csv"))
    tags = DataFrame(CSV.File(dataset * "/tags.csv"))

    return links, movies, ratings, tags
end
