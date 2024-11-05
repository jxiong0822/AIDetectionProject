

echo Running test.bat

echo Unigram
python main.py -f unigram > unigram.txt

echo Bigram
python main.py -f bigram > bigram.txt

echo Trigram
python main.py -f trigram > trigram.txt