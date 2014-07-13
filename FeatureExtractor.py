class Extractor:
    def __init__(self):
        self.LoadBrownFreq()
        self.LoadSUBTLEXusFreq()
        self.LoadScrabbleScore()
    
    def LoadBrownFreq(self):
        from nltk.corpus import brown
        from nltk import FreqDist
        #ref: http://www.nltk.org/book/ch02.html
        text = brown.words()
        self.brownFreqDist = FreqDist([w.lower() for w in text])
    
    def LoadSUBTLEXusFreq(self, path="./data/SUBTLEXus74286.txt"):
        f = open(path)
        for line in f.readlines():
            lineSpl = line.split('\t')
            
        
        
        

if __name__=="__main__":
    print "Hello"
    