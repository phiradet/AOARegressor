from FeatureExtractor import DataShuffler, Regressor, Evaluator
from FeatureExtractor import AOAReader
from sklearn.cross_validation import KFold

def BasicCrossValidation():
    K=5
    words, aoaVals = AOAReader()
    kf = KFold(len(aoaVals), n_folds=K, shuffle=True)
    k=1
    for trainIndices, testIndices in kf:
        print "FOLD ",k
        print "#train", len(trainIndices), "\t#test", len(testIndices)
        trainFeature, trainLabel, testFeature, testLabel = DataShuffler.SplitData(words, aoaVals, trainIndices, testIndices)
        
        r = Regressor()
        r.Learn(trainFeature, trainLabel)
        predictedVal = r.Predict(testFeature)
        
        print "MSE", Evaluator.MSE(testLabel, predictedVal)
        print "################################"
        k+=1

if __name__=="__main__":
    BasicCrossValidation()
        
        
        
        
        
