import json

file_path = './dev-v2.0.json'

with open(file_path, 'r') as json_file:
    squad = json.load(json_file)

sentence_over_100 = []

#a = 0

for i in range(0,20):
    for j in range(0,10):
        #a += 1
        #print(len(squad['data'][i]['paragraphs'][j]['context']))
        if (len(squad['data'][i]['paragraphs'][j]['context']) >= 500):
            sentence_over_100.append(squad['data'][i]['paragraphs'][j]['context'][:500])

#print(len(sentence_over_100[1]))
file_out_path = './concatenated_500_sentences.json'
with open(file_out_path, 'w') as json_file:
    json.dump(sentence_over_100[:100], json_file)
print('------------done  ')


