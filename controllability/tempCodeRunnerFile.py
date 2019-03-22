# Load sample DTI data
test_filename = '/Users/skyeong/pythonwork/controllability/data/testgraph.edgelist'
G=nx.read_edgelist(test_filename)
nodelist = [str(i+1) for i in range(82)]
A=nx.to_numpy_matrix(G,nodelist=nodelist)