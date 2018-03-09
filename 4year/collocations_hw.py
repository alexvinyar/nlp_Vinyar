from nltk import collocations
from scipy.stats import spearmanr


gs = [('ПРИНЯТЬ', 'РЕШЕНИЕ'),('УДОВЛЕТВОРИТЬ', 'ИСК'),('УДОВЛЕТВОРИТЬ', 'ХОДАТАЙСТВО'),
      ('САНКЦИЯ', 'АРЕСТ'),('САНКЦИОНИРОВАТЬ', 'АРЕСТ'),('ВЫНЕСТИ', 'РЕШЕНИЕ'),('ИЗБРАТЬ', 'МЕРА'),
      ('ПРИЗНАТЬ', 'ВИНОВНАЯ'),('НАЛОЖИТЬ', 'АРЕСТ'),('ОТКАЗАТЬ', 'УДОВЛЕТВОРЕНИЕ')]


words = []
with open('court-V-N.csv','r',encoding='utf-8') as f:
    for line in f.readlines():
        words.append(line.strip().split(' ,'))

bigram_measures = collocations.BigramAssocMeasures()
finder = collocations.BigramCollocationFinder.from_documents(words)
finder.apply_freq_filter(5)

print('PMI')
for i in finder.nbest(bigram_measures.pmi, 10):
    print(i)

print('\nLikelihood ratio')
for i in finder.nbest(bigram_measures.likelihood_ratio, 10):
    print(i)


all_pmi = [x[0] for x in finder.score_ngrams(bigram_measures.pmi)]
all_lr = [x[0] for x in finder.score_ngrams(bigram_measures.likelihood_ratio)]

ranks_pmi = [all_pmi.index(x)+1 for x in gs]
ranks_lr = [all_lr.index(x)+1 for x in gs]

print('\nCorrelation between PMI and Likelihood ratio')
print(spearmanr(ranks_pmi,ranks_lr))
print('\nCorrelation between PMI and Golden Standard')
print(spearmanr(ranks_pmi,list(range(1,11))))
print('\nCorrelation between Likelihood ratio and Golden Standard')
print(spearmanr(ranks_lr,list(range(1,11))))
