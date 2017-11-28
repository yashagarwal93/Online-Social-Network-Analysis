"""
cluster.py
"""
import pickle
from pathlib import Path
import sys
import time
from TwitterAPI import TwitterAPI
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
import copy
import math
from collect import get_twitter,open_file


def find_unique_user(tweets):
    user=[]
    user_id=[]
    for i in tweets:
        user.append(i['screen_name'])
        user_id.append(i['userid'])
    return list(set(user)),list(set(user_id))



def draw_graph(names):
    """
    Create the  graph from ids of the followers.
    """
    teams = list(names.keys())
    g = nx.Graph()
    for i in range(len(teams)):
        for j in range(i+1,len(teams)):
            a =list(set(names[teams[i]] ) & set(names[teams[j]]))
            for node in a:
                g.add_edges_from([(teams[i],node)])
                g.add_edges_from([(teams[j],node)])
                #g.add_edges_from([(teams[i],teams[j])])

    return g

#get_ipython().magic('matplotlib inline')
def read_graph(g,names):

    labels = {}
    pos = nx.spring_layout(g)
    plt.figure(figsize=(100,100))
    nx.draw(g, pos, with_labels=False,node_size =2500)
    for node in g.nodes():
        for i in names:
            if i == node:
                labels[node] = i

    nx.draw_networkx_labels(g,pos,labels=labels,font_size=50,font_color='b')
    plt.savefig("before_clusture.png")
    return g

def girvan_newman(G,minsize, maxsize):

    """
    Args:
    G.........a networkx graph
    minsize...the smallest acceptable size for a community
    maxsize...the largest acceptable size for a community
    Returns:
    A list of all discovered communities. This list is a list
    of lists of strings. E.g., [['A', 'B'], ['C', 'D']] is a result
    containing two communities ((A,B) and (C,D)).
    """
    if G.order() == 1:
        return [G.nodes()]

    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)

        return sorted(eb.items(), key=lambda x: x[1], reverse=True)



    components = [c for c in nx.connected_component_subgraphs(G)]
    edge_to_remove = find_best_edge(G)
    count = 0
    while len(components) == 1:
        G.remove_edge(*edge_to_remove[count][0])
        count = count +1
        components = [c for c in nx.connected_component_subgraphs(G)]


    result = []
    for c in components:
        if (c.order()>=minsize and c.order()<=maxsize):
            result.append(c.nodes())
        elif c.order()>maxsize:
            result.extend(girvan_newman(c,minsize,maxsize))
    return result


def draw_network(graph, users, f):

    ###TODO
    name="Cluster"+str(f)+".png"
    pos=nx.spring_layout(graph)
    plt.figure(figsize=(50,50))
    nx.draw(graph,pos,with_labels=False,node_size =3000)
    labels = {}

    for node in graph.nodes():
        for i in users:
            if i == node:
                labels[node] = i

    nx.draw_networkx_labels(graph,pos,labels=labels,font_size=55,font_color='b')
    plt.savefig(name)
    pass


def write_file(tweets):
    fname='Cluster.pkl'
    print("Cluster Information stored in file")
    pickle.dump(tweets, open(fname, 'wb'))


def main():
    twitter = get_twitter()
    tweets=[]
    names={}
    hashtags=['#Celtics','#DetroitBasketball','#DefendTheLand','#WeTheNorth','#MADEinPHILA']
    text_file = open("cluster.txt" , "w")

    for tags in hashtags:
        text_file1=[]
        count=0
        tweets= open_file(tweets,tags)
        tweeter_names,tweeter_id=find_unique_user(tweets)
        count+=len(tweeter_names)
        names[tags]=tweeter_names
    teams = list(names.keys())
    for i in range(len(teams)):
        for j in range(i+1,len(teams)):
                a =list(set(names[teams[i]] ) & set(names[teams[j]]))
                #print ("Mutual number of followers for the team %s %s ---->" %teams[i] %teams[j],len(a))
                print("Mutual number of followers for the team",teams[i],teams[j],"---->",len(a))
                with open('cluster.txt', 'a') as f:
                    print("Mutual number of followers for the team",teams[i],teams[j],"---->",len(a),file=f)
                """text_file1.append("Mutual number of followers for the team:")
                text_file1.append(teams[i])
                text_file1.append(teams[j])
                text_file1.append("---->")
                text_file1.append(len(a))
                text_file.write(text_file1)
                #print(text_file1)
                """

    graph=draw_graph(names)
    g=read_graph(graph,names)
    print('graph has %d nodes and %d edges' %(graph.order(), graph.number_of_edges()))
    with open('cluster.txt', 'a') as f:
        print('graph has %d nodes and %d edges' %(graph.order(), graph.number_of_edges()),file=f)
    text_file1.append('graph has %d nodes and %d edges' %(graph.order(), graph.number_of_edges()))
    result=girvan_newman(g,5, 20)
    name=1
    TEMP={}
    for i in range(1,len(result)):
        with open('cluster.txt', 'a') as f:
            print ('cluster %d  Number of nodes/followers %d' %(i ,len(result[i])),file=f)
        print ('cluster %d  Number of nodes/followers %d' %(i ,len(result[i])))
        draw_network(graph.subgraph(result[i]),result[i],i)
        TEMP[i]=result[i]
        TEMP['usercount']=count
        with open('cluster.txt', 'a') as f:
            print("Cluster %d nodes" %i ,result[i], file=f)
    write_file(TEMP)


if __name__ == '__main__':
    main()
