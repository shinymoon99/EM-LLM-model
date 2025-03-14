"""
Evaluate the predictive ability of blocks
1、Determine if it contains the correct function name
2、Judging whether it is legal
3、compute ISM and PM
"""
import json
import tokenize
import io
import math
import ast

def is_code_valid(code):

    try:
        compile(code, '<string>', 'exec')
        return True
    except:
        return False


def longest_common_prefix_between_lists_with_elements(list1, list2):
    """
    Calculate the longest prefix matching length of elements in two string lists
    :param list1:
    :param list2:
    :return:
    """
    max_prefix_length = 0
    max_prefix_elements = ()
    for str1 in list1:
        for str2 in list2:
            prefix_length = 0
            min_len = min(len(str1), len(str2))
            for i in range(min_len):
                if str1[i] == str2[i]:
                    prefix_length += 1
                else:
                    break
            if prefix_length > max_prefix_length:
                max_prefix_length = prefix_length
                max_prefix_elements = (str1, str2)
    return max_prefix_length, max_prefix_elements

def get_token(ans_code:str, output_code:str):
    """
    Perform lexical analysis on the code, decompose it into identifiers, and return two lists of identifiers
    :param ans_code:
    :param output_code:
    :return:
    """
    output_flag = True
    ans_flag = True
    try:
        tokens_ans = tokenize.tokenize(io.BytesIO(ans_code.encode('utf-8')).readline)
    except Exception as e:
        tokens_ans = ans_code.splitlines()
        ans_flag = False

    try:
        tokens_output = tokenize.tokenize(io.BytesIO(output_code.encode('utf-8')).readline)
    except Exception as e:
        tokens_output = output_code.splitlines()
        output_flag = False


    identifiers_ans = []
    identifiers_output = []
    if ans_flag == True:
        try:
            for token in tokens_ans:
                if token.type == tokenize.NAME:
                    identifiers_ans.append(token.string)
        except Exception as e:
            identifiers_ans = tokens_ans
    else:
        identifiers_ans = tokens_ans

    if output_flag == True:
        try:
            for to in tokens_output:
                if to.type == tokenize.NAME:
                    identifiers_output.append(to.string)
        except Exception as e:
            identifiers_output = tokens_output
    else:
        identifiers_output = tokens_output


    return identifiers_ans, identifiers_output


def get_token_per_line(code: str):
    """
    Perform lexical analysis on each line of code and record the identifier for each line
    :param code: code string
    :return: A list composed of identifier lists for each row
    """
    lines = code.split('\n')
    identifiers_per_line = []

    for line in lines:
        tokens = tokenize.tokenize(io.BytesIO(line.encode('utf-8')).readline)
        identifiers = []
        try:
            for token in tokens:
                if token.type == tokenize.NAME:
                    identifiers.append(token.string)
        except:
            identifiers = line.split(' ')
        identifiers_per_line.append(identifiers)

    return identifiers_per_line



def get_ISM(answer_code:str, model_output_list:list, asnwer_name:str)->list:
    """
    compute ISM, return an ordered list of scores
    :return:
    """
    score_list = []
    for code in model_output_list:

        # if asnwer_name not in code or is_code_valid(code) == False:
        #     score_list.append(0)
        #     continue



        identifiers_ans, identifiers_output = get_token(answer_code, code)
        max_len, elements = longest_common_prefix_between_lists_with_elements(identifiers_ans, identifiers_output)
        if max_len != 0:
            base_element_len = max(len(elements[0]), len(elements[1]))
            temp_score = max_len/base_element_len
            score_list.append(temp_score)
        else:
            score_list.append(0)

    score_list = sorted(score_list, reverse=True)
    return score_list

def get_ISM_without_verification(answer_code:str, model_output_list:list, asnwer_name:str)->list:
    """
    copmpute ISM, return an ordered list of scores
    :return:
    """
    score_list = []
    for code in model_output_list:

        if asnwer_name not in code:
            score_list.append(0)
            continue

        # if asnwer_name not in code:
        #     score_list.append(0)
        #     continue

        identifiers_ans, identifiers_output = get_token(answer_code, code)
        max_len, elements = longest_common_prefix_between_lists_with_elements(identifiers_ans, identifiers_output)
        if max_len != 0:
            base_element_len = max(len(elements[0]), len(elements[1]))
            temp_score = max_len/base_element_len
            score_list.append(temp_score)
        else:
            score_list.append(0)

    score_list = sorted(score_list, reverse=True)
    return score_list

def longest_common_prefix_with_lengths(list1, list2):
    """
    Calculate the longest prefix matching length for each sublist in two two-dimensional lists, and record the length of the two sublists with the longest prefix matching length
    :param list1:
    :param list2:
    :return:
    """
    max_length = 0
    len_list1 = 0
    len_list2 = 0
    for i, sublist1 in enumerate(list1):
        for j, sublist2 in enumerate(list2):
            match_length = 0
            min_length = min(len(sublist1), len(sublist2))
            for k in range(min_length):
                if sublist1[k] == sublist2[k]:
                    match_length += 1
                else:
                    break
            if match_length > max_length:
                max_length = match_length
                len_list1 = len(sublist1)
                len_list2 = len(sublist2)
    return max_length, len_list1, len_list2


def get_PM(answer_code:str, model_output_list:list, asnwer_name:str)->list:
    """
    compute PM，return an ordered list of scores
    :return:
    """

    score_list = []
    for code in model_output_list:
        # if asnwer_name not in code or is_code_valid(code) == False:
        #     score_list.append(0)
        #     continue



        ans_list = get_token_per_line(answer_code)
        output_token_list = get_token_per_line(code)
        max_len, len1, len2 = longest_common_prefix_with_lengths(ans_list, output_token_list)
        base_element_len = max(len1, len2)

        if base_element_len != 0:
            temp_score = max_len/base_element_len
            score_list.append(temp_score)
        else:
            score_list.append(0)

    score_list = sorted(score_list, reverse=True)
    return score_list

def get_score(score_list:list, k):
    """
    compute score@n,k
    :param score_list:
    :param k:
    :return:
    """
    n = len(score_list)
    sum = 0
    final = n-k+1
    for i in range(1, final+1):
        sum += math.comb(n-i, k-1) * score_list[i-1]

    final_score = sum/math.comb(n, k)

    return final_score




# block_json_path = '../../dataset/final_dataset/code_editing_result_new_to_old/Llama-3-70b-chat-hf/test_without_docstring.json'   #code editing new to old的结果
block_json_path = 'output/versicode_RAG.json'
with open(block_json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_list = lodict
data_len = len(data_list)
sum_ISM = 0
sum_PM = 0

for data in data_list:
    model_output_list = [data['answer']] #change block or token or line
    # model_output_list = data['model_output_block_clear']
    temp_list = []
    for o in model_output_list:
        # 获取<start>和<end>之间的代码
        temp_out = o.split('<start>')[1].split('<end>')[0]
        temp_out = temp_out.replace('```python', '')
        temp_out = temp_out.replace('```', '')
        temp_list.append(temp_out)
    model_output_list = temp_list
    answer_code = data['ground_truth']    #common block
    answer_name = data['id']  #common block(与该项无关，所以随便给了个值)

    # answer_code = data['new_code']  #code editing
    # answer_name = data['new_name']    #code editing

    # answer_code = data['old_code']  # code editing new to old
    # answer_name = data['old_name']  # code editing new to old
    print("answer_code: ",answer_code)
    print("answer_name: ",answer_name)
    print("model_output_list: ",model_output_list)
    # ISM_score_list = get_ISM(answer_code, model_output_list, answer_name)
    # PM_score_list = get_PM(answer_code, model_output_list, answer_name)
    ISM_score_list = get_ISM(answer_code, model_output_list, answer_name)
    PM_score_list = get_PM(answer_code, model_output_list, answer_name)
    print("ISM_score_list:", ISM_score_list)
    print("PM_score_list:", PM_score_list)

    if not ISM_score_list or not PM_score_list:
        print("Warning: Empty score list for:")
        print("answer_code:", answer_code)
        print("answer_name:", answer_name)
        print("model_output_list:", model_output_list)
        continue  # Skip this iteration if either score list is empty

    #TODO:看看以下get_score如何使用
    ISM_score = get_score(ISM_score_list, 1)
    PM_score = get_score(PM_score_list, 1)

    sum_ISM += ISM_score
    sum_PM += PM_score

print(f"ISM：{sum_ISM/data_len}")
print(f"PM：{sum_PM/data_len}")



