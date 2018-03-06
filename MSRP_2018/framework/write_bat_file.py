'''
    start: python adjust_para.py 2 2 1000 1000 >> adjust_para.result
'''
with open(r'..\framework\run.bat', 'w', encoding='utf-8') as fw:
    for filter_size in range(2, 6):
        for top_k in range(1, 11):
            # for full_cell in(1000,5000):
                    fw.writelines('start: python adjust_para.py ' + str(filter_size) + ' ' + str(top_k) + ' ' +
                                  str(5000) + ' ' + str(10000) + ' >> adjust_para.result\n')
    fw.close()
