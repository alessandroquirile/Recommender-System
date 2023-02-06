using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV
using DataFrames

#Poissibili valori: ml-latest, ml-latest-small, ml-100k, ml-1m, ml-10m, ml-20m, ml-25m
dataset = "ml-latest-small"
dataset_zip = dataset * ".zip" #dove scaricare il file

#Se il file non è già presente, scaricalo e decomprimilo
if !isfile(dataset_zip)
    url = "https://files.grouplens.org/datasets/movielens/" * dataset_zip
    download(url, dataset_zip)
    run(`unzip $dataset_zip`)
end

#Carica la tabella ratings e mostrane le prime 6 righe
ratings = DataFrame(CSV.File(dataset * "/ratings.csv"))
println(first(ratings, 6))

