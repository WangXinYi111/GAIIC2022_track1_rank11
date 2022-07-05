import json

#  输入5折预测结果路径
file1 = 'data/tmp_data/textcnn_result1.txt'
file2 = 'data/tmp_data/textcnn_result2.txt'
file3 = 'data/tmp_data/textcnn_result3.txt'
file4 = 'data/tmp_data/textcnn_result4.txt'
file5 = 'data/tmp_data/textcnn_result5.txt'
file6 = 'data/tmp_data/Bert_result1.txt'
file7 = 'data/tmp_data/Bert_result2.txt'
# 输出路径
out_file = "data/submission/results.txt"
submit = []
#  得到一行数据
def get_diff(key):
    out_key = key[0]
    for j,value in key[0]['match'].items():
        label1 = 0
        label0 = 0
        for i in range(7):
            if key[i]['match'][j]==1:
                label1 += 1
            if key[i]['match'][j]==0:
                label0 += 1
        if label1>label0:
            out_key['match'][j]=1
        if label1<label0:
            out_key['match'][j]=0
    submit.append(json.dumps(out_key, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    with open(file1, 'r', encoding='utf-8') as f:
        line1 = f.readlines()
    with open(file2, 'r', encoding='utf-8') as f:
        line2 = f.readlines()
    with open(file3, 'r', encoding='utf-8') as f:
        line3 = f.readlines()
    with open(file4, 'r', encoding='utf-8') as f:
        line4 = f.readlines()
    with open(file5, 'r', encoding='utf-8') as f:
        line5 = f.readlines()
    with open(file6, 'r', encoding='utf-8') as f:
        line6 = f.readlines()
    with open(file7, 'r', encoding='utf-8') as f:
        line7 = f.readlines()
        for i in range(len(line2)):
            key =[]
            key1 = json.loads(line1[i])
            key2 = json.loads(line2[i])
            key3 = json.loads(line3[i])
            key4 = json.loads(line4[i])
            key5 = json.loads(line5[i])
            key6 = json.loads(line6[i])
            key7 = json.loads(line7[i])
            key = [key1,key2,key3,key4,key5,key6,key7]
            get_diff(key)
with open(out_file, 'w',encoding='utf-8') as f:
    f.writelines(submit)




