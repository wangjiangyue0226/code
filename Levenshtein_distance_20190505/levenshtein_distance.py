#-*-coding：utf-8-*-

"""
Levenshtein 距离的python实现
Levenshtein 距离，又称编辑距离，指的是两个字符串之间，由一个转换成另一个所需的最少编辑操作次数。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。
"""


def levenshtein_distance(first_word, second_word):
    """
    :param first_word: 第一个字符串
    :param second_word: 第二个字符串
    :return: 返回两个字符串间的Levenshtein 距离
    Examples:
    >>> levenshtein_distance("planet", "planetary")
    3
    >>> levenshtein_distance("", "test")
    4
    >>> levenshtein_distance("book", "back")
    2
    >>> levenshtein_distance("book", "book")
    0
    >>> levenshtein_distance("test", "")
    4
    >>> levenshtein_distance("", "")
    0
    >>> levenshtein_distance("orchestration", "container")
    10
    """

    if len(first_word) < len(second_word):
        return levenshtein_distance(second_word, first_word)

    if len(second_word) == 0:
        return len(first_word)

    previous_row = range(len(second_word) + 1)

    for i, c1 in enumerate(first_word):

        current_row = [i + 1]

        for j, c2 in enumerate(second_word):

            # 计算插入,删除和替换次数
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)


            current_row.append(min(insertions, deletions, substitutions))


        previous_row = current_row


    return previous_row[-1]


if __name__ == '__main__':
    try:
        raw_input          # Python 2
    except NameError:
        raw_input = input  # Python 3

    first_word = raw_input('Enter the first word:\n').strip()
    second_word = raw_input('Enter the second word:\n').strip()

    result = levenshtein_distance(first_word, second_word)
    print('Levenshtein distance between {} and {} is {}'.format(
        first_word, second_word, result))
