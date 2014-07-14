from nltk_contrib.readability import syllables_en
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import cross_validation

class Extractor:
    def __init__(self):
        self.LoadBrownFreq()
        self.LoadSUBTLEXusFreq()
        self.LoadScrabbleScore()
        self.LoadScrabbleDist()
        self.VOWELS = ('a', 'e', 'i', 'o', 'u')
    
    def LoadBrownFreq(self):
        from nltk.corpus import brown
        from nltk import FreqDist
        #ref: http://www.nltk.org/book/ch02.html
        text = brown.words()
        self.BrownFreqeuncy = FreqDist([w.lower() for w in text])
    
    def LoadSUBTLEXusFreq(self, path="./data/SUBTLEXus74286.txt"):
        self.SUBTLEFreqDist = dict()
        f = open(path)
        for line in f.readlines()[1:]:
            lineSpl = line.split('\t')
            word = lineSpl[0].lower()
            freq = float(lineSpl[1])
            freqInM = float(lineSpl[5])
            if word in self.SUBTLEFreqDist.keys():
                self.SUBTLEFreqDist[word][0]+=freq
                self.SUBTLEFreqDist[word][1]+=freqInM
            else:
                self.SUBTLEFreqDist[word] = (freq, freqInM)
        f.close()
    
    def LoadScrabbleScore(self, path="./data/ScrabbleScore"):
        self.Char2ScScore = dict()
        self.ScScore2Char = dict()
        f = open(path)
        for line in f.readlines():
            lineSpl = line.strip().split(':')
            score = float(lineSpl[0])
            chars = map(str.lower, lineSpl[1].split('\t'))
            self.ScScore2Char[score]=chars
            for c in chars:
                self.Char2ScScore[c] = score
        f.close()
    
    def LoadScrabbleDist(self, path="./data/ScrabbleDist"):
        self.Char2ScDist = dict()
        self.ScDist2Char = dict()
        f = open(path)
        for line in f.readlines():
            lineSpl = line.strip().split(':')
            dist = float(lineSpl[0])
            chars = map(str.lower, lineSpl[1].split('\t'))
            self.ScDist2Char[dist] = chars
            for c in chars:
                self.Char2ScDist[c] = dist
    
    def VowelSegment(self, word):
        import re
        m = re.match("(.*?)[a|e|i|o|u](.*)", word)
        if bool(m):
            return m.groups()
        else:
            return (word,'')
    
    def CalculateScScore(self, word):
        score = 0
        if len(word)==0:
            return 0
        for c in word:
            score += self.Char2ScScore[c]
        return score
    
    def CalculateScDist(self, word):
        dist = 0
        if len(word)==0:
            return 0
        for c in word:
            try:
                dist += self.Char2ScDist[c]
            except KeyError:
                print self.Char2ScDist
                raise "HEY"
        return dist
    
    def VowelSegmentCount(self, word):
        return map(len, self.VowelSegment(word))
    
    ####################################
    ########## FEATURE TABLE ###########
    
    #Feature 1: Word Lenth ex. eat=3, school=6
    def WordLength(self, word):
        return len(word)
    
    #Feature2: Syllable Count, ex passport=2, good=1
    def SyllableCount(self, word):
        return syllables_en.count(word)
    
    #Feature 3: #of characters before the first vowel
    def BeforeVowelCount(self, word):
        before, after = self.VowelSegment(word)
        return len(before)
        
    #Feature 4: #of characters after the first vowel
    def AfterVowelCount(self, word):
        before, after = self.VowelSegment(word)
        return len(after)
    
    #Feature 5: Scrabble score of characters before the first vowel
    def BeforeVowelScScore(self, word):
        before, after = self.VowelSegment(word)
        beforeScore = self.CalculateScScore(before)
        return beforeScore
    
    #Feature 6: Scrabble score of characters after the first vowel
    def AfterVowelScScore(self, word):
        before, after = self.VowelSegment(word)
        afterScore = self.CalculateScScore(after)
        return afterScore
    
    #Feature 7: Scrabble score of the first character
    def ScScoreOf1stChar(self, word):
        firstChar = list(word)[0]
        return self.CalculateScScore(firstChar)
    
    #Feature 8: Scrabble score of the last character
    def ScScoreOfLastChar(self, word):
        lastChar = list(word)[-1]
        return self.CalculateScScore(lastChar)
    
    #Feature 9: Scrabble score of word
    def ScScoreOfWord(self, word):
        return self.CalculateScScore(word)
    
    #Feature 10: Scrabble Dist of word
    def ScDistOfWord(self, word):
        return self.CalculateScDist(word)
    
    #Feature 11: Freq in Brown Corpus
    def GetBrownFreqeuncy(self, word):
        freq = self.BrownFreqeuncy[word]
        return freq
    
    #Feature 12: Freq in SUBTLEXus Corpus
    def GetSUBTLEXFrequency(self, word):
        freq = self.SUBTLEFreqDist.get(word,(0,0))[0]
        #print word, freq
        return freq
    
    #Feature 13,14,15,16,17,18,19: ratio of character in each type scrabble group
    def ScrabbleScoreRatio(self, word):
        scoreKey = self.ScScore2Char.keys()
        scoreKey.sort()
        output = [0]*len(scoreKey)
        for c in word:
            currScore = self.Char2ScScore[c]
            scoreIndex = scoreKey.index(currScore)
            output[scoreIndex]+=1
        return map(lambda x:float(x)/float(len(word)),output)
    
    #Feature 20, 21: Number of vowels and consonant
    def NumOfVowelAndCon(self, word):
        vowelsNum = 0.
        constNum = 0.
        for c in list(word):
            if c not in self.VOWELS:
                constNum+=1.
            else:
                vowelsNum+=1.
        wordLen = float(len(word))
        return [vowelsNum/wordLen,constNum/wordLen]
    
    def CreateFeatureRawVector(self, word):
        monoExtractor = [self.WordLength, self.SyllableCount, self.BeforeVowelCount,
                                self.AfterVowelCount, self.BeforeVowelScScore, self.AfterVowelScScore,
                                self.ScScoreOf1stChar, self.ScScoreOfLastChar, self.ScScoreOfWord,
                                self.ScDistOfWord, self.GetBrownFreqeuncy, self.GetSUBTLEXFrequency]
        multiExtractor = [self.ScrabbleScoreRatio, self.NumOfVowelAndCon]
        
        featureVector = []
        for monoExt in monoExtractor:
            value = monoExt(word)
            featureVector.append(value)
        for mulExt in multiExtractor:
            values = mulExt(word)
            featureVector += values
        return featureVector
    
    def CreateFeatureTable(self, words, isTrain=False, AVGs=None, SDs=None):
        import pickle
        rawFeatureTable = []
        for w in words:
            row = self.CreateFeatureRawVector(w) 
            rawFeatureTable.append(row)
            #print row
        rawFeatureTableNp = np.matrix(rawFeatureTable)
        if isTrain:
            AVGs, SDs = self.GetStandardScoreParam(rawFeatureTableNp)
            pickle.dump((AVGs, SDs), open("./data/normParam.bin","wb"))
            normFeatureTable = self.StandardizeTable(rawFeatureTableNp, AVGs, SDs)
            return normFeatureTable
        else:
            return self.StandardizeTable(rawFeatureTableNp, AVGs, SDs)
    
    def StandardScoreVector(self, vector, AVGs, SDs):
        """
            vector is list
        """
        normVector = []
        for i in range(len(vector)):
            val = vector[i]
            newVal = (val-AVGs[i])/(SDs[i])
            normVector.append(newVal)
        return normVector
    
    def StandardizeTable(self, table, AVGs, SDs):
        rNum, cNum = table.shape
        normTable = None
        for c in range(cNum):
            f = np.vectorize(lambda x:(x-AVGs[c])/SDs[c], otypes=[np.float])
            normCol = f(table[:,c])
            if normTable==None:
                normTable = normCol
            else:
                normTable = np.concatenate((normTable, normCol), axis=1)
        #=======================================================================
        # print normTable.shape
        # print normTable
        # print table
        # print SDs
        # print AVGs
        # print  "==============="
        #=======================================================================
        return normTable
            
    def GetStandardScoreParam(self, table):
        """
            table (numpy matrix)
        """
        rNum,cNum = table.shape
        AVGs = []
        SDs = []
        for col in range(cNum):
            currAVG = np.mean(table[:,col])
            currSD = np.std(table[:,col])
            AVGs.append(currAVG)
            SDs.append(currSD)
        return AVGs, SDs

class Regressor:
    def __init__(self):
        pass
    
    def Learn(self, words,AOAs, verbose=True):
        ext = Extractor()
        trainFeatureTable = ext.CreateFeatureTable(words, isTrain=True)
        self.clf = linear_model.Ridge()
        self.clf.fit (trainFeatureTable, AOAs)
        if verbose:
            print "coef"
            for coef in self.clf.coef_:
                print coef
    
    def Predict(self, words):
        import pickle
        AVGs, SDs = pickle.load(open("./data/normParam.bin","rb"))
        ext = Extractor()
        testFeatureTable = ext.CreateFeatureTable(words, False, AVGs, SDs)
        predictedResult = self.clf.predict(testFeatureTable)
        return predictedResult
    
class Evaluator:
    @staticmethod
    def MSE(yActual, yPredict):
        mse = mean_squared_error(yActual, yPredict)
        return mse
    
    @staticmethod
    def R2(yActual, yPredict):
        r2 = r2_score(yActual, yPredict) 
        return r2        

class DataShuffler:
    @staticmethod
    def RandomSplit(size):
        rs = cross_validation.ShuffleSplit(n=size, n_iter=1, test_size=.25, random_state=0)
        return rs
    
    @staticmethod
    def SplitData(features, labels, trainIndex, testIndex):
        trainFeature = []
        trainLabel = []
        for i in trainIndex:
            trainFeature.append(features[i])
            trainLabel.append(labels[i])
        
        testFeature = []
        testLabel = []
        for i in testIndex:
            testFeature.append(features[i])
            testLabel.append(labels[i])
        return trainFeature, trainLabel, testFeature, testLabel            

def AOAReader(path="./data/aoa.csv"):
    f = open(path)
    words = []
    aoaVals = []
    for line in f.readlines():
        lineSpl = line.split('\t')
        if lineSpl[1].strip()=='NA':
            continue
        word = lineSpl[0].strip().lower().replace('-','')
        words.append(word)
        aoaVal = float(lineSpl[1].strip())
        aoaVals.append(aoaVal)
    return words, aoaVals
    
def demo():
    words, aoaVals = AOAReader()
    dataSize = len(words)
    splitter = DataShuffler.RandomSplit(size=dataSize)
    for tr,te in splitter:
        pass
    trainFeature, trainLabel, testFeature, testLabel = DataShuffler.SplitData(words, aoaVals, tr, te)
    #-------------------------------------------------------- print trainFeature
    #---------------------------------------------------------- print trainLabel
    #--------------------------------------------------------- print testFeature
    #----------------------------------------------------------- print testLabel
    r = Regressor()
    r.Learn(trainFeature, trainLabel)
    predictedVal = r.Predict(testFeature)
    print predictedVal

def TestFeatureExtrator():
    ext = Extractor()
    monoExtractor = [ext.WordLength, ext.SyllableCount, ext.BeforeVowelCount,
                        ext.AfterVowelCount, ext.BeforeVowelScScore, ext.AfterVowelScScore,
                        ext.ScScoreOf1stChar, ext.ScScoreOfLastChar, ext.ScScoreOfWord,
                        ext.ScDistOfWord, ext.GetBrownFreqeuncy, ext.GetSUBTLEXFrequency]
    multiExtractor = [ext.ScrabbleScoreRatio, ext.NumOfVowelAndCon]
    word = 'school'
    print 'word:',word
    for m in monoExtractor:
        print m.__name__, m(word),

if __name__=="__main__":
    np.set_printoptions(threshold=25)
    #demo()
    #Test()
    
