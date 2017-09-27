#!/bin/sh
# created by danny
# the function to create the dictionary is based on a script by:
#  Eleanor Chodroff on 2/22/15.
# This script filters out words which are not in our corpus.
# It requires a list of the words in the corpus: words.txt

# this prepares a lexicon for kaldi
# it starts from an existing lexicon but to use it in kaldi we need
# to remove words that are not in your current database. 
# furthermore we might need to add some words that are in your data
# base but not in the existing lexicon. However the phonetic transcripts of cgn
# are not checked so a chance of errors exists and it is best to use these transcripts
# only to complete an existing lexicon.

from label_func import parse_transcript
from data_functions import list_files
import codecs
from lex import create_lexicon

# set an out of vocab token. The 'ortographic' and 'phonetic' version used here are the same, they do not have to be.
oov = ('<oov>', '<oov>') 

# path to the cgn awd transcripts
datapath="/home/danny/Downloads/comp-o/nl"

# path to existing lexicon
lex_loc="/home/danny/Downloads/Finetracker/kaldi/lexicon.txt"

# path to save the new lexicon. Your lexicon should end up in your
# kaldi projects data/local/lang folder
new_lex_loc="/home/danny/Downloads/Finetracker/kaldi/lexicon_comp_o.txt"

# location of our phonelist. be sure to run the script kaldi_phones.py first to create this list
phones_loc="/home/danny/Downloads/Finetracker/kaldi/nonsilence_phones.txt"

#pattern to retrieve ortographic transcript 
pattern= "N[0-9]+"

# at several points we strip some punctuation and special characters from the transcript 
striplist= '_.,?!=\"'

# list input files and sort (not necesarry but it might make it easier to manually
# look in your files if something goes wrong)
input_files = list_files(datapath)
input_files.sort()

# remove durational info, headers etc using the parse_transcript function.
trans=[parse_transcript(pattern,datapath+"/"+y) for y in input_files]
words=[]

for tr in trans:
    # pick with step size 3 (skip durational info in CGN format) 
    for x in range (2,len(tr),3):
        # split the transcript (awd transcripts should already be split but somehow in CGN some
        # sentences are not properly split yet)
        split_tr = tr[x].split()
        # add the words to the word list and replace some special tokens
        for word in split_tr:
            word= word.replace(u'\xb1',u'plusminus').replace(u'\xd7',u'').replace(u'\xb3',u'').replace(u'–',u' ').replace(u'$', u'dollar').replace(u'%',u'procent').replace(u' & ',u' en ').replace(u'&amp',u'en').replace(u'&',u'en').replace(u'\x90',u' ').replace(u'\x91',u' ').replace(u'\x92',u' ').replace(u'\x93',u' ').replace(u'\x94',u' ').replace(u'\x95',u' ').replace(u'\x96',u' ').replace(u'\x97',u' ').replace(u'\x98',u' ').replace(u'\x99',u' ').replace(u'\xbd',u'').replace(u'\xff',u'').replace(u'\u2663',u'').replace(u'\u2666',u'').replace(u'\u2660',u'').replace(u'\u2665',u'').replace(u'\xb9',u'').replace(u'\xb2',u'').replace(u'\u2070',u'').replace(u'\u2079',u'').replace(u'\u2074',u'').replace(u'\u0660',u'').replace(u'\u2075',u'').replace(u'\u2071',u'').replace(u'\u2072',u'').replace(u'\u2073',u'').replace(u'\u2076',u'').replace(u'\u2077',u'').replace(u'\u2078',u'').replace(u'\u2792',u'').replace(u'\u2082',u'').replace(u"1/2","half").replace(u"/",u" ").replace(u'~',u'')         
            # ggg, xxx and Xxx equal unintelligeble parts which could not be transcribed
            # we replace these with an out of vocab token
            if not 'ggg' in word and not 'xxx' in word and not 'Xxx' in word:
                words.append(word)
            else:
                words.append(oov[0])
# strip punctuation
words = [str.lower(x.strip(striplist))for x in words if len(x.strip(striplist))>0]

# remove some cgn annotation for foreign and dialect words (e.g. *v and *u which are added to the end of the word)
for x in range (0,len(words)):
    if '*' in words[x]:
        words[x]=words[x][:-2]

# remove doubles and sort
words = list(set(words))
words.sort()

ref = dict()
# open and load kaldi formatted lexicon into a dictionary
with codecs.open(lex_loc, "rb") as f:
    for line in f:
        # the pre-made lexicon I found and used somehow is in part encoded in utf-8
        # and part in iso-2. luckily utf throws errors if something is not encoded
        # in utf so it easy to detect and use iso-2 where appropriate
        try:
            line=line.decode('utf-8')
        except:
            line=line.decode('iso-8859-2')

        line = line.strip()
        # split into ortographic form and pronunciation. A kaldi formatted lexicon should 
	# always contain lines of format <word>space<pronunciation> pronunciation should 
        # have whitespace between phonemes
        columns = line.split(None, 1)
        word = columns[0]
        pronunciation = columns[1]
        # add words to dict and map multiple pronunciations to a word
        try:
            ref[word].append(pronunciation)
        except:
            ref[word] = list()
            ref[word].append(pronunciation) 
           
# get the list of phones which occur in your database
phone_list=[]
nonsilence_phones = codecs.open(phones_loc, "rb")
for line in nonsilence_phones:
    line=line.decode('utf-8')
    line=line.strip()
    phone_list.append(line)         

# now we need to filter words that are not in the database from the lexicon. 
# kaldi will complain if your lexicon contains words that do not occur in the training data
new_ref={}
for word in words:   
    if word in ref.keys():
        new_ref[word]=list()
        for pronunciation in ref[word]:
            # check to see if the phones used in the pronunciation are used in our database
            check = [x for x in pronunciation.split() if not x in phone_list]
            # if check > 0 the pronuncation contains unwanted phones and we do not add it to our 
            # lexicon
            if len(check)==0:
                new_ref[word].append(pronunciation)

# there might be words occuring in your data which are not yet in the premade lexicon
# we get a list of those words so we can complement the premade lexicon.
out_of_vocab=[]
for x in words:
    try: 
        new_ref[x]
    except:
        out_of_vocab.append(x)
# function I wrote to create a lexicon from awd transcripts
ref= create_lexicon(datapath, oov)  
# for all the out of vocabulary words, add the versions from our own lexicon       
for ov in out_of_vocab:      
    pronunciation = ref[ov]
    try:
        for pronun in pronunciation:
            new_ref[ov].append(pronun)
    except:
        new_ref[ov]= list()
        for pronun in pronunciation:
            new_ref[ov].append(pronun)  
        
# check if the dictionary is of equal length with words, e.g. all words in our
# data are now represented in the lexicon
print ('word list and lexicon of equal length: '+ str(len(new_ref)== len (words)))  

# lastly, in the premade lexicon I used it seems some pronunciations for the same word 
# occur twice, leading to kaldi errors. remove these doubles.
for word in new_ref:
     new_ref[word] = list(set(new_ref[word]))

# make a new lexicon in the appropriate kaldi folder (in utf-8)    
lex = codecs.open(new_lex_loc, "w")
# add an out of vocab word phone pair to the lexicon. Even though
# we ensured our lexicon contains all words in the data, kaldi needs this option
# as a fallback
line = oov[0]+ ' '+ oov[1]+'\n'
lex.write(line)
# since we already add an oov token by default, pop it from the dict in case its already in the lexicon
try:
    new_ref.pop(oov[0])
except:
    pass
x = [x for x in new_ref]
x.sort()
for word in x:
    for pronunciation in new_ref[word]:
        line  = word+ " " + pronunciation+"\n"
        line.encode('utf-8')
        lex.write(line)
lex.close()
