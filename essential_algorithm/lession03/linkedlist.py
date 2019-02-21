#!/usr/bin/python3

class Node:
    def __init__(self,data,next=None):
        self.data = data
        self.next = next

def init_list():
    global node_A
    node_A = Node("A")
    node_B = Node("B")
    node_D = Node("D")
    node_E = Node("E")
    node_A.next = node_B
    node_B.next = node_D
    node_D.next = node_E

def insert_node(data):
    global node_A
    new_node = Node(data)
    node_P = node_A
    node_T = node_A
    while node_T.data <= data:
        node_P = node_T
        node_T = node_T.next
    new_node.next = node_T
    node_P.next = new_node

def print_list():
    global node_A 
    node = node_A
    while node:
        print(node.data)
        node = node.next
    print

if __name__=="__main__":
    print("Initialize nodes")
    init_list()
    print_list()
    print("Inserting Node C")
    insert_node("C")
    print_list()