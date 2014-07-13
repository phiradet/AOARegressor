from nltk_contrib.readability import syllables_en
import numpy as np

class Extractor:
    def __init__(self):
        self.LoadBrownFreq()
        self.LoadSUBTLEXusFreq()
        self.LoadScrabbleScore()
        self.VOWELS = ('a', 'e', 'i', 'o', 'u')
    
    def LoadBrownFreq(self):
        from nltk.corpus import brown
        from nltk import FreqDist
        #ref: http://www.nltk.org/book/ch02.html
        text = brown.words()
        self.brownFreqDist = FreqDist([w.lower() for w in text])
    
    def LoadSUBTLEXusFreq(self, path="./data/SUBTLEXus74286.txt"):
        self.SUBTLEFreqDist = dict()
        f = open(path)
        for line in f.readlines():
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
            lineSpl = line.split(':')
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
            lineSpl = line.split(':')
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
        for c in word:
            score += self.Char2ScScore[c]
        return score
    
    def CalculateScDist(self, word):
        dist = 0
        for c in word:
            dist += self.Char2ScDist[c]
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
    def  BeforeVowelCount(self, word):
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
        after = self.VowelSegment(word)
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
    def BrownFreqeuncy(self, word):
        freq = self.BrownFreqeuncy.get(word,0)
        return freq
    
    #Feature 12: Freq in SUBTLEXus Corpus
    def SUBTLEXFrequency(self, word):
        freq = self.SUBTLEXFrequency.get(word,0)
        return freq
    
    #Feature 13,14,15,16,17,18,19: ratio of character in each type scrabble group
    def ScrabbleScoreRatio(self, word):
        scoreKey = self.ScScore2Char.keys()
        scoreKey.sort()
        output = [0]*
        for c in word:
            currScore = self.Char2ScScore[c]
            scoreIndex = scoreKey.index(currScore)
            output[scoreIndex]+=1
        return map(lambda x:float(x)/float(len(word)),output)
    
    #Feature 20, 21: Number of vowels and consonant
    def NumberOfVowelAndCon(self, word):
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
                                self.BeforeVowelCount, self.BeforeVowelScScore, self. AfterVowelScScore,
                                self.ScScoreOf1stChar, self.ScScoreOfLastChar, self.ScScoreOfWord
                                self.ScDistOfWord, self.BrownFreqeuncy, self.SUBTLEXFrequency]
        multiExtractor = [ScrabbleScoreRatio, NumOfVowelAndCon]
        
        featureVector = []
        for monoExt in monoExtractor:
            value = monoExt(word)
            featureVector.append(value)
        for mulExt in multiExtractor:
            values = multiExtractor(word)
            featureVector += values
        return featureVector
    
    def CreateFeatureTable(self, words, isNorm=False, AVGs=None, SDs=None):
        import picke
        rawFeatureTable = []
        for w in words:
            rawFeatureTable.append(self.CreateFeatureRawVector(w))
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
        normTable = np.matrix()
        for c in range(cNum):
            f = np.vectorize(lambda x:(x-AVGs[c])/SDs[c], otypes=[np.float])
            normCol = f(table[:,c])
            normTable = np.concatenate((normTable, normCol), axis=1)
        return normTable
            
    def GetStandardScoreParam(self, table):
        """
            table (numpy matrix)
        """
        rNum,cNum = table.shape
        AVGs = []
        SDs = []
        for c in range(cNum):
            currAVG = np.mean(table[:,col])
            currSD = np.std(table[:,col])
            AVGs.append(currAVG)
            SDs.append(currSD)
        return AVGs, SDs

class Regressor:
    def __init__(self):
        pass
    
    def Learn(self, words,AOAs):
        pass
    
    def Predict(self, words):
        pass
    
if __name__=="__main__":
    print "Hello"
    
