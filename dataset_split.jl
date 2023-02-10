function datasetSplit(ratingsDataFrame, foldSize, foldIndex)
    numberOfRatings = size(ratingsDataFrame, 1)
    numberOfTestSetRatings = floor(Int, numberOfRatings * foldSize)

    testSetStartIndex = foldIndex * numberOfTestSetRatings + 1 # +1 because Julia's indices start from 1
    testSetEndIndex = testSetStartIndex + numberOfTestSetRatings - 1 # -1 because Julia intervals are inclusive

    indices = range(1, numberOfRatings)
    foldIndices = indices[testSetStartIndex:testSetEndIndex]
    trainingSetIndices = filter(x -> x ∉ foldIndices, indices)

    foldDataFrame = ratingsDataFrame[ foldIndices, :]
    trainingSetDataFrame = ratingsDataFrame[ trainingSetIndices, :]

    return trainingSetDataFrame, foldDataFrame
end