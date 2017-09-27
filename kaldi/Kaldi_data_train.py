#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:55:41 2017

@author: emre

augmented by danny
This file creates some necessary files for kaldi. In doing so 
we replace some CGN notations, for instance 'Xxx' for unintelligible words
with '<oov>' , '%' with 'procent' and '$' with 'dollar'.  Furthermore we strip CGN *v and *u (to indicate foreign and dialect words).
PLEASE NOTE, these replacements NEED to be consistent with the replacements
in kaldi_lexicon.py and lex.py. If you change something here, change it there as well or
your segments file and lexicon will not match, kaldi does not like that and if kaldi
does not like something you will have a very bad day. 

"""
import pdb, os, glob, codecs, re, argparse, random

#parser = argparse.ArgumentParser(description='Lexicon preparation')
#parser.add_argument('--speaker_adapt', help='indicates speaker dependent or independent recognition', type=str, required=True)

#args = parser.parse_args()
# set mode (SA is speaker adaptive SI speaker independent)
speaker_info = 'SA'
# folder to put data in
data_folder = '/home/danny/Downloads/temp/'

print('CGN DUTCH DATA')

suffix='train'
speaker_list={}
speaker_cnt=0
unk_cnt=0

# folder paths for your wav files
main_folder_wav = '/home/danny/Downloads/temp/'
# folder paths for your transcripts. Please note
# this script works now specifically for ort (ortographic) transcription
# files of CGN. If your files are differently formatted you might need to 
# change some parts of the code 
main_folder_annot = '/home/danny/Downloads/'
train_annotated_data=['comp-o']

new_txt = ''
new_seg = ''
new_utt = ''
new_wav = ''

fid_txt = codecs.open(data_folder+suffix+'/text',"w","utf-8")
fid_seg = codecs.open(data_folder+suffix+'/segments',"w","utf-8")
fid_utt = codecs.open(data_folder+suffix+'/utt2spk',"w","utf-8")
fid_wav = codecs.open(data_folder+suffix+'/wav.scp',"w","utf-8")

for task in train_annotated_data:
    init_trans = main_folder_annot + task
    filelist=glob.glob(init_trans+"/nl/*.ort")
    for filename in filelist:
        cnt3=0
        cnt2=0
        temp = filename.split('/')
        utt_id = temp[-1][:-4]
        flag=0
        flag2=0
        new_wav = new_wav+task+'_'+utt_id+' '+main_folder_wav+task+'/nl/'+utt_id+'.wav'+'\n'
        cnt=4
        speaker_flag=0

        for line in codecs.open(filename,'r','iso-8859-1'):
            # These checks are specific to CGN formatting, Intervaltier is simply
            if u'"IntervalTier"' in line:
                cnt2=cnt2+1
                speaker_flag=1
                flag=0
                cnt=4
                continue
            if u'"BACKGROUN' in line or u'"COMMEN' in line:
                break
            if speaker_flag==1:
                speaker_code=line[1:-3].lower()              
                if speaker_code==u'unknown':
                    speaker= 'cu'+(str("{:04.0f}".format(unk_cnt)))
                    if speaker not in speaker_list.keys():
                        speaker_list[speaker] = 'cu'+(str("{:04.0f}".format(unk_cnt)))
                        unk_cnt=unk_cnt+1                               
                    speaker=speaker_list[speaker]
                    speaker_flag=0
                else:
                    speaker= 'cg'+speaker_code
                    if speaker not in speaker_list.keys():
                        speaker_list[speaker] = 'cg'+(str("{:04.0f}".format(speaker_cnt)))
                        speaker_cnt=speaker_cnt+1
                    speaker=speaker_list[speaker]
                    speaker_flag=0

            if cnt2<1:
                continue
            if cnt>0 and flag==0:
                cnt=cnt-1
                flag=0
                continue
            else:
                flag=1
                cnt=cnt+1           
            if cnt%3==1:
                s_time = (str("{:.3f}".format(float(line))))
                continue
            if cnt%3==2:
                e_time = (str("{:.3f}".format(float(line))))
                continue
            if cnt%3==0:
                if u'""' in line or float(e_time)-float(s_time) < 0.15 or u'"nsp"' in line:
                    continue
                text=line
                # place here any characters that need to be replaced by whitespace or for instance replace % sign with 'procent'. 
                text=text.replace(u'\xb1',u'plusminus').replace(u'\xd7',u'').replace(u'\xb3',u'').replace(u'–',u' ').replace(u'$', u'dollar').replace(u'%',u'procent').replace(u' & ',u' en ').replace(u'&amp',u'en').replace(u'&',u'en').replace(u'\x90',u' ').replace(u'\x91',u' ').replace(u'\x92',u' ').replace(u'\x93',u' ').replace(u'\x94',u' ').replace(u'\x95',u' ').replace(u'\x96',u' ').replace(u'\x97',u' ').replace(u'\x98',u' ').replace(u'\x99',u' ').replace(u'\xbd',u'').replace(u'\xff',u'').replace(u'\u2663',u'').replace(u'\u2666',u'').replace(u'\u2660',u'').replace(u'\u2665',u'').replace(u'\xb9',u'').replace(u'\xb2',u'').replace(u'\u2070',u'').replace(u'\u2079',u'').replace(u'\u2074',u'').replace(u'\u0660',u'').replace(u'\u2075',u'').replace(u'\u2071',u'').replace(u'\u2072',u'').replace(u'\u2073',u'').replace(u'\u2076',u'').replace(u'\u2077',u'').replace(u'\u2078',u'').replace(u'\u2792',u'').replace(u'\u2082',u'').replace(u"1/2","half").replace(u"/",u" ").replace(u'~',u'')
                text = u"".join(c for c in text if c not in  (u'!',u'.',u':',u'?',u',',u'\n',u'\r',u'"',u'|',u';',u'(',u')',u'[',u']',u'{',u'}',u'#',u'_',u'+',u'&lt',u'&gt',u'\\'))
                fields = text.lower().split()
                for ele in fields:
                    # place here words that need to be replaced. for instance replace stop words such as uhm, eh, uh with a garbage phoneneme such as spn 
                    # I placed here a few lines that detects CGN specific annotation like *v and *a for dialect and foreign words, but instead of mapping them to a garbage phoneme 
                    # I simply remove the annotation and keep the words
                    if u'*' in ele: 
                        ind=fields.index(ele)
                        fields[ind]=fields[ind][:-2]
                    elif u'ggg' in ele or u'xxx' in ele:
                        ind=fields.index(ele)
                        fields[ind]=u'<oov>'
                text = u' '.join(fields)
                temp=text.strip().lower().split()
                text = u' '.join(temp)
                # different formatting for speaker adaptive (SA) and speaker independent (SI) modes
                if speaker_info=='SA' and text!=u'spn' and text!=u'' and text!=u'nsn':
                    new_txt = new_txt+speaker+u'_'+task+u'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u' '+text+u'\n'
                    new_seg = new_seg+speaker+u'_'+task+u'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u' '+task+u'_'+utt_id+u' '+s_time+u' '+e_time+u'\n'
                    new_utt = new_utt+speaker+u'_'+task+u'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u' '+speaker+u'\n'
                elif speaker_info=='SI' and text!=u'spn' and text!=u'' and text!=u'nsn':
                    new_txt = new_txt+task+u'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u' '+text+u'\n'
                    new_seg = new_seg+task+u'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u' '+task+'_'+utt_id+u' '+s_time+u' '+e_time+u'\n'
                    new_utt = new_utt+task+u'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u' '+task+'_'+utt_id+u'_'+(str("{:04.0f}".format(cnt3)))+u'\n'
                cnt3=cnt3+1
            if cnt3%300:
                fid_txt.write(new_txt)
                fid_seg.write(new_seg)
                fid_utt.write(new_utt)
                fid_wav.write(new_wav)
                new_txt = ''
                new_seg = ''
                new_utt = ''
                new_wav = ''
fid_txt.write(new_txt)
fid_seg.write(new_seg)
fid_utt.write(new_utt)
fid_wav.write(new_wav)
fid_txt.close()
fid_seg.close()
fid_utt.close()
fid_wav.close()

