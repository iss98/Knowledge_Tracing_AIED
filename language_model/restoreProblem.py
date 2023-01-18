def restoreProblem(encText, encEqns, textTokenizer, mathTokenizer):

    mathNum = textTokenizer.encode('[MATH]')[1]
    restoration = str()
    mathCount = 0
    for t in encText:
        if t == mathNum:
            restoration += mathTokenizer.decode(encEqns[mathCount]).replace('[CLS]', '').replace('[SEP]', '').replace(' [PAD]', '')
            mathCount +=1
        else:
            restoration += textTokenizer.decode(t).replace('[CLS]', '').replace('[SEP]', '').replace(' [PAD]', '').replace('#', '')

    return restoration