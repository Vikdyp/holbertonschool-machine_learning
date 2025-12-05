#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(people))

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(x, apples, width=0.5, color='red', label='apples')
    plt.bar(x, bananas, width=0.5, bottom=apples,
            color='yellow', label='bananas')
    plt.bar(x, oranges, width=0.5, bottom=apples + bananas,
            color='#ff8000', label='oranges')
    plt.bar(x, peaches, width=0.5, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='peaches')

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()

bars()