import matplotlib.pyplot as plt
import csv
import sys

if __name__=='__main__':
    # data_to_plot = 'Average Score' if len(sys.argv) < 2 else sys.argv[2]
    x = []
    score = []
    fruit1 = []
    fruit2 = []
    fruit3 = []
    task_completion = []
    with open(sys.argv[1],'r') as csvfile:
        data = csv.DictReader(csvfile, delimiter=',')
        for row in data:
            x.append(int(row['Epoch']) / 10000)
            score.append(float(row[' Average Score']) / 100)
            fruit1.append(float(row[' Fruit 1%']))
            fruit2.append(float(row[' Fruit 2%']))
            fruit3.append(float(row[' Fruit 3%']))
            task_completion.append(int(row[' Num Task Completion']))

    plt.plot(x,score)
    plt.title('Score averaged over intervals of 10,000 iterations across training')
    plt.xlabel('Epoch (x10^5)')
    plt.ylabel('Score (x100)')
    plt.show()

    plt.plot(x, fruit1, color='r', label='Fruit 1%')
    plt.plot(x, fruit2, color='g', label='Fruit 2%')
    plt.plot(x, fruit3, color='b', label='Fruit 3%')
    plt.legend()
    plt.xlabel('Epoch (x10^5)')
    plt.ylabel('% out of 10k iterations wherein fruit was picked')
    plt.title('Percentage of times out of 10k iterations where a particular fruit was picked across training')
    plt.show()

    plt.plot(x, task_completion)
    plt.xlabel('Epoch (x10^5)')
    plt.ylabel('Count (/10k) when task has been completed')
    plt.title('Number of times out of 10k iterations when the task has been completed')
    plt.show()
