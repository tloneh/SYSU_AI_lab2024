from GeneticAlgTSP import *

def main():
    filename = 'test.txt'
    TSP = GeneticAlgTSP(filename)
    num = 100
    TSP.iterate(num)

if __name__ == '__main__':
    main()
