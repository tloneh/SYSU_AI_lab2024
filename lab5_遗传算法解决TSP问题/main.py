from GeneticAlgTSP import *

def main():
    filename = 'test.txt'
    TSP = GeneticAlgTSP(filename)
    num = 3000 # 取得一个较优结果
    TSP.iterate(num)

if __name__ == '__main__':
    main()
