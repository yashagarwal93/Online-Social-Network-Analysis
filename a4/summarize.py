"""
sumarize.py
"""
import os

"""
def open_clusture(fname):
    p = Path(fname)
    if p.is_file():
        with open(fname, "rb") as file:
            try:
                tweets = pickle.load(file)
            except EOFError:
                return tweets
    return tweets
"""


def read_method_details():
    """
    This method reads the details saved in files that occurs when you run Collect.py, Cluster.py and Classify.py
    and prints the details to screen and save it to "summary.txt" .
    :return: Nothing
    """

    with open("collect.txt",'r' ) as fp:
         collector_details = fp.read()
    fp.close()
    with open("cluster.txt",'r') as gp:
        cluster_details = gp.read()
    gp.close()
    """
    with open('before_clusture.png', 'r') as i1:
        cluster_image = i1.read()
    i1.close()
    with open('before clustering', 'r') as i2:
        cluster_image = i1.read()
    i1.close()
    """
    with open("classify.txt",'r') as dp:
        classify_details = dp.read()
    dp.close()

    with open("summary.txt",'w') as sp:
        sp.write("Collect.py Details : \n")
        sp.write(collector_details)
        sp.write("Cluster.py Details : \n")
        sp.write(cluster_details)
        sp.write("Classify.py Details : \n")
        sp.write(classify_details)

    sp.close()

    print("Details saved to --> summary.txt")

    return collector_details,cluster_details,classify_details


def main():
    """
    clusname='Cluster.pkl'
    c_details=open_clusture(clusname)
    for i  in range(1,5):
        clus_count.append(len(c_details[i]))

    numberofuser= c_details['usercount']
    """

    print("\t\t************************ - Starting summary.py - ************************ ")

    collector_details,cluster_details,classify_details  = read_method_details()
    print("\n")
    print("\t\t---- Collect_Details ----\n")
    print(collector_details)
    print("\n")
    print("\t\t---- Cluster_Details ----\n")
    print(cluster_details)
    print("\n")
    print("\t\t---- Classifier_Details ----\n")
    print(classify_details)
    print("\n")
    print("\n\t\t************************ - Finished Printing Summary - ************************")


if __name__ == main():
    main()
