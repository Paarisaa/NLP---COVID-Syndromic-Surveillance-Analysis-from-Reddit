# -*- coding: utf-8 -*-
"""
Assignment1

@author: psarikh
"""

import nltk
import re
import pandas as pd
import xlsxwriter
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import Levenshtein
import math

# Read the input data:
Unlabeled_data = pd.read_excel('./UnlabeledSet2.xlsx')

# create the output xlsx file:
output_file_name = 'Output_Assignment1GoldStandardSet_Parisa_Sarikhani.xlsx'
workbook = xlsxwriter.Workbook(output_file_name)
worksheet = workbook.add_worksheet()
worksheet.write(0,0, 'ID')
worksheet.write(0,1, 'Symptom CUIs')
worksheet.write(0,2, 'Negation Flag')

# Create a symptom dictionary for the COVID-Twitter-Symptom-Lexicon: 
infile_symp = open('./COVID-Twitter-Symptom-Lexicon.txt', encoding='utf-8')
symptom_text = infile_symp.readlines()
symptom_dict = {}
for line in symptom_text:
    symptom_dict[line.split('\t')[2][:-1]] = line.split('\t')[1]
    
# Add symptom expression from my annotations to the symptom dictionary:    
my_annotations = pd.read_excel('s9.xlsx')
my_annotations = my_annotations.fillna('nan')
for iteration, expression in my_annotations.iterrows():
    if expression['Symptom Expressions']!='nan':
        espression_split = expression['Symptom Expressions'].split('$$$')
        sym_cui_split = expression['Symptom CUIs'].split('$$$')
        for sym, cui in zip(espression_split, sym_cui_split):
            if sym and cui:
                symptom_dict[sym] = cui 




# read neg_trigs.txt file
neg_infile = open('./neg_trigs.txt')
neg_trigs = neg_infile.readlines()
neg_trigs_list = [neg[:-1] for neg in neg_trigs] # negation list

# preprocessing function:
def preprocessing(input_text):
    processed_input = sent_tokenize(input_text.lower())
    return processed_input

print('-------------analysing the gold standrad file--------------------------')
'''
Using the Gold Standard File'''

# Read the texts and ids of the gold standrad file to a pd dataframe  (will use just the ids and texts)
Gold_standard = pd.read_excel('./Assignment1GoldStandardSet.xlsx')
Gold_standard = Gold_standard.fillna('nan')
  
# go through each text
for it, item in Gold_standard.iterrows():
    if item['TEXT']!='nan':
        preprocessed_post = preprocessing(item['TEXT'])
        worksheet.write(it+1,0,item['ID']) # write ID into the excel worksheet
        sympt_cui = '$$$'
        neg_flag = '$$$'
        for sent in preprocessed_post: # for each sentence
            # word tokenized version of each sentence
            wrd_tokenized = list(word_tokenize(sent))
            bigrams = list(nltk.ngrams(wrd_tokenized,n=3))  #ngrams of the sentence
            trigrams = list(nltk.ngrams(wrd_tokenized,n=4))  #ngrams of the sentence
            
            for symp in symptom_dict.keys(): # for each symptom
                pat = re.compile(r'\b'+symp+r'\b') # regular expression
                iter_object = re.finditer(pat,sent) # find all symptom matches in each sentence
                
                
                if iter_object:
                    
                    for match in iter_object:
                        previous_words = list(nltk.word_tokenize(sent[:match.start()])) 
                        three_prev_words = previous_words[-3:] # find the three previous words of the symptom to find the negation
                        if any([item in three_prev_words for item in neg_trigs_list]) or '.' in three_prev_words:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag += '1$$$'
                            # print (sent+'\t'+symptom_dict[symp]+'-neg'+'\n')
                            # print('===========',previous_words[-3:])
                        else:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag +='0$$$'
                            # print (sent+'\t'+symptom_dict[symp]+'\n')
                    worksheet.write(it+1, 1, sympt_cui)
                    worksheet.write(it+1, 2, neg_flag)
                    
            
            # Add ngrams and fuzzy matching
            
                for ngram in bigrams:
                    ngram_str = ' '.join(ngram)
                    # print(ngram_str)
                    ratio = Levenshtein.ratio(str.strip(ngram_str),symp)
                    # print(ratio)
                    if ratio>0.95:
                        match_start = sent.find(ngram_str)# find the start position of the ngram
                        previous_words = list(nltk.word_tokenize(sent[:match_start])) 
                        three_prev_words = previous_words[-3:] # find the three previous words of the symptom to find the negation
                        if any([item in three_prev_words for item in neg_trigs_list]) or '.' in three_prev_words:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag += '1$$$'
                        else:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag +='0$$$'
                            
                            
                            
                for ngram in trigrams:
                    ngram_str = ' '.join(ngram)
                    # print(ngram_str)
                    ratio = Levenshtein.ratio(str.strip(ngram_str),symp)
                    # print(ratio)
                    if ratio>0.95:
                        match_start = sent.find(ngram_str)# find the start position of the ngram
                        previous_words = list(nltk.word_tokenize(sent[:match_start])) 
                        three_prev_words = previous_words[-3:] # find the three previous words of the symptom to find the negation
                        if any([item in three_prev_words for item in neg_trigs_list]) or '.' in three_prev_words:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag += '1$$$'
                        else:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag +='0$$$'
                            
        worksheet.write(it+1, 1, sympt_cui)
        worksheet.write(it+1, 2, neg_flag)

# close the output file
workbook.close()

print('-------------Evaluating the gold standrad file--------------------------')
import pandas as pd
from collections import defaultdict



def load_labels(f_path):
    '''
    Loads the labels

    :param f_path:
    :return:
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(list)
    for index,row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['Symptom CUIs']) and not pd.isna(row['Negation Flag']):
            cuis = row['Symptom CUIs'].split('$$$')[1:-1]
            neg_flags = row['Negation Flag'].split('$$$')[1:-1]
            for cui,neg_flag in zip(cuis,neg_flags):
                labeled_dict[id_].append(cui + '-' + str(neg_flag))
    return labeled_dict

gold_standard_dict = load_labels('./Assignment1GoldStandardSet.xlsx')
# submission_dict = load_labels('./AssignmentSampleSubmission.xlsx')
submission_dict = load_labels('./Output_Assignment1GoldStandardSet_Parisa_Sarikhani.xlsx')

tp = 0
tn = 0
fp = 0
fn = 0
for k,v in gold_standard_dict.items():
    for c in v:
        try:
            if c in submission_dict[k]:
               tp+=1
            else:
                fn+=1
        except KeyError:#if the key is not found in the submission file, each is considered
                        #to be a false negative..
            fn+=1
    for c2 in submission_dict[k]:
        if not c2 in gold_standard_dict[k]:
            fp+=1
print('True Positives:',tp, 'False Positives: ', fp, 'False Negatives:', fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1 = (2*recall*precision)/(recall+precision)
print('Recall: ',recall,'\nPrecision:',precision,'\nF1-Score:',f1)



print('-------------analysing the unlabeled reddit posts--------------------------')
'''
Using the unlabeled reddit posts'''

# Read the input data:
Unlabeled_data = pd.read_excel('./UnlabeledSet2.xlsx')
Unlabeled_data = Unlabeled_data.fillna('nan')
# create the output xlsx file:
output_file_name = 'Output_Submission_Reddit_Posts_Parisa_Sarikhani.xlsx'
workbook = xlsxwriter.Workbook(output_file_name)
worksheet = workbook.add_worksheet()
worksheet.write(0,0, 'ID')
worksheet.write(0,1, 'Symptom CUIs')
worksheet.write(0,2, 'Negation Flag')



  
# go through each text
for it, item in Unlabeled_data.iterrows():
    if item['TEXT']!='nan':
        preprocessed_post = preprocessing(item['TEXT'])
        worksheet.write(it+1,0,item['ID']) # write ID into the excel worksheet
        sympt_cui = '$$$'
        neg_flag = '$$$'
        for sent in preprocessed_post: # for each sentence
            # word tokenized version of each sentence
            wrd_tokenized = list(word_tokenize(sent))
            bigrams = list(nltk.ngrams(wrd_tokenized,n=3))  #ngrams of the sentence
            trigrams = list(nltk.ngrams(wrd_tokenized,n=4))  #ngrams of the sentence
            
            for symp in symptom_dict.keys(): # for each symptom
                pat = re.compile(r'\b'+symp+r'\b') # regular expression
                iter_object = re.finditer(pat,sent) # find all symptom matches in each sentence
                
                
                if iter_object:
                    
                    for match in iter_object:
                        previous_words = list(nltk.word_tokenize(sent[:match.start()])) 
                        three_prev_words = previous_words[-3:] # find the three previous words of the symptom to find the negation
                        if any([item in three_prev_words for item in neg_trigs_list]) or '.' in three_prev_words:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag += '1$$$'
                            # print (sent+'\t'+symptom_dict[symp]+'-neg'+'\n')
                            # print('===========',previous_words[-3:])
                        else:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag +='0$$$'
                            # print (sent+'\t'+symptom_dict[symp]+'\n')
                    # worksheet.write(it+1, 1, sympt_cui)
                    # worksheet.write(it+1, 2, neg_flag)
                    
            
            # Add ngrams and fuzzy matching
            
                for ngram in bigrams:
                    ngram_str = ' '.join(ngram)
                    # print(ngram_str)
                    ratio = Levenshtein.ratio(str.strip(ngram_str),symp)
                    # print(ratio)
                    if ratio>0.95:
                        match_start = sent.find(ngram_str)# find the start position of the ngram
                        previous_words = list(nltk.word_tokenize(sent[:match_start])) 
                        three_prev_words = previous_words[-3:] # find the three previous words of the symptom to find the negation
                        if any([item in three_prev_words for item in neg_trigs_list]) or '.' in three_prev_words:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag += '1$$$'
                        else:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag +='0$$$'
                            
                            
                            
                for ngram in trigrams:
                    ngram_str = ' '.join(ngram)
                    # print(ngram_str)
                    ratio = Levenshtein.ratio(str.strip(ngram_str),symp)
                    # print(ratio)
                    if ratio>0.95:
                        match_start = sent.find(ngram_str)# find the start position of the ngram
                        previous_words = list(nltk.word_tokenize(sent[:match_start])) 
                        three_prev_words = previous_words[-3:] # find the three previous words of the symptom to find the negation
                        if any([item in three_prev_words for item in neg_trigs_list]) or '.' in three_prev_words:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag += '1$$$'
                        else:
                            sympt_cui += symptom_dict[symp]+'$$$'
                            neg_flag +='0$$$'
        worksheet.write(it+1, 1, sympt_cui)
        worksheet.write(it+1, 2, neg_flag)

# close the output file
workbook.close()

