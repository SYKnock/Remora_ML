from pororo import Pororo
import os
root = os.getcwd()


def ngram(sentence, num):
    result = []
    total_len = len(sentence) - num + 1
    for i in range(total_len):
        tmp = sentence[i:i + num]
        result.append(tmp)
    return result


def difference_check_ngram(sentence_1, sentence_2, num):
    ngram_1 = ngram(sentence_1, num)
    ngram_2 = ngram(sentence_2, num)
    cnt = 0
    for char_1 in ngram_1:
        for char_2 in ngram_2:
            if char_1 == char_2:
                cnt += 1
    return cnt / len(ngram_1)


def recognition(path):
    buffer_dir = root + path
    ocr = Pororo(task="ocr", lang="ko")
    dirs = os.listdir(buffer_dir)
    dirs = sorted(dirs, key=lambda x: int(x))

    script = ""
    check = ""
    for directory in dirs:
        buffer_dir_detail = buffer_dir + "/" + directory
        target_list = os.listdir(buffer_dir_detail)
        target_list = sorted(target_list, key=lambda x: int(x[0:-4]))
        sentence = ""
        for img in target_list:
            img_path = buffer_dir_detail + "/" + img
            print(img_path)
            character = ocr(img_path)

            if not character:
                continue
            
            if not sentence:
                sentence += character[0]
            else:
                sentence = sentence + " " + character[0]

        if not sentence:
            continue

        if not script:
            script += sentence
            check = sentence
        else:
            # roughly check it first
            check_token = sentence.split()
            score = 0
            for char in check_token:
                if check.find(char) != -1:
                    score += 1
            if score / len(check_token) > 0.7:
                continue
            else:
                # ngram check
                if difference_check_ngram(check, sentence, 3) > 0.7:
                    continue
                else:
                    check = sentence
                    script = script + "\n" + sentence

    return script