function kFoldSplit(ratingsDataFrame, testSetSize, kFoldIndex)
    numberOfRatings = size(ratingsDataFrame, 1)
    numberOfTestSetRatings = floor(Int, numberOfRatings * testSetSize)

    testSetStartIndex = kFoldIndex * numberOfTestSetRatings + 1 # +1 because Julia's indices start from 1
    testSetEndIndex = testSetStartIndex + numberOfTestSetRatings

    indices = range(1, numberOfRatings)
    testSetindices = indices[testSetStartIndex:testSetEndIndex]
    trainingSetindices = filter(x -> x ∉ testSetindices, indices)

    testSetDataFrame = ratingsDataFrame[ testSetindices, :]
    trainingSetDataFrame = ratingsDataFrame[ trainingSetindices, :]

    return trainingSetDataFrame, testSetDataFrame
end