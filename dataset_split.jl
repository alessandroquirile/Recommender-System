"""
Costruisce una URM dividendo il datset in training e test

"""
function kFoldSplit(ratingsDataFrame, numberOfUsers, numberOfMovies, testSetSize, kFoldIndex)
    numberOfRatings = size(ratingsDataFrame, 1)
    numberOfTestSetRatings = floor(Int, numberOfRatings * testSetSize)

    testSetStartIndex = kFoldIndex * numberOfTestSetRatings + 1 # +1 because Julia's indexes start from 1
    testSetEndIndex = testSetStartIndex + numberOfTestSetRatings

    indexes = range(1, numberOfRatings)
    testSetIndexes = indexes[testSetStartIndex:testSetEndIndex]
    trainingSetIndexes = filter(x -> x ∉ testSetIndexes, indexes)

    testSetDataFrame = ratingsDataFrame[ testSetIndexes, :]
    trainingSetDataFrame = ratingsDataFrame[ trainingSetIndexes, :]

    return trainingSetDataFrame, testSetDataFrame
end