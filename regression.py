import numpy as np
from ca import make_argument_parser as ca_make_arg_parser, Data, UpdateRule
from trainer import Trainer, basic_train
import ca
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy import spatial

class RegressionTrainer(Trainer):
    def __init__(self, args):
        pass

    def train(self, partitions, graph):
        num_cities = len(graph)
        num_pairs = sum((len(x)-2) * num_cities for x in partitions)
        y = np.zeros(num_pairs)     
        x = np.zeros(shape = (num_pairs, 4))
        i = 0
        for fold in partitions:
            for t, time in enumerate(fold):
                if t < 2:
                    continue
                else:
                    for c, city in enumerate(time):
                        y[i] = city
                        x[i][0] = fold[t-1][c]
                        x[i][1] = fold[t-2][c]
                        x[i][2] = np.dot(fold[t-1], graph[c])
                        x[i][3] = np.dot(fold[t-2], graph[c])
                        i += 1

        self.model = linear_model.LinearRegression()
        self.model.fit(x,y)

        w = np.append(self.model.coef_, self.model.intercept_)
        w = np.reshape(w, (1,5))
        w2 = np.array([[1,0,0,0,0]])
        w = np.concatenate((w, w2), axis =0)
        
        self.update_rule = ca.UpdateRule(graph, w)
        return self.update_rule

    def test(self, partition, graph):
        prediction = np.zeros(shape = np.shape(partition))
        for t, time in enumerate(partition):
            if t < 2:
                continue
            else:
                for c, city in enumerate(time):
                    x = np.zeros(5)
                    x[0] = partition[t-1][c]
                    x[1] = partition[t-2][c]
                    x[2] = np.dot(partition[t-1], graph[c])
                    x[3] = np.dot(partition[t-2], graph[c])
                    x[4] = 1
                    prediction[t][c] = np.dot(x, self.update_rule.weights[0])
        pearson = np.zeros(len(graph))
        cosine_similarity = np.zeros(len(graph))

        for city_index in range(len(graph)):
            city_actual = partition[2:,city_index]
            city_predicted = prediction[2:,city_index]
            pearson[city_index] = stat.pearsonr(city_actual, city_predicted)[0]
            cosine_similarity[city_index] = 1 - spatial.distance.cosine(city_actual, city_predicted)            
            plt.plot(range(len(partition) -2), city_actual, label = 'Actual')
            plt.plot(range(len(partition) -2), city_predicted, label = 'Predicted')
            plt.ylabel('Count of Flu-Related Searches')
            plt.xlabel('Time')
            plt.legend(loc = 'best')
            plt.title("City " + str(city_index) + "'s Flu-Related Search Trends")       
            print("Saving output for city " + str(city_index))
            plt.savefig("output/RegressionCA_City" + str(city_index) + "_Actual_vs_Predicted.png")
            plt.clf()   
                
        plt.plot(range(len(graph)), pearson, linestyle = 'None', marker = r'$\bowtie$')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.xlabel('Index of City')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, len(graph))
        plt.title("Pearson Coefficient Between Actual\nand Predicted Counts for All Cities")
        plt.savefig("output/RegressionCA_PearsonCoef.png")
        plt.clf()

        plt.plot(range(len(graph)), cosine_similarity, linestyle = 'None', marker = r'$\bowtie$')
        plt.ylabel('Cosine Similarity')
        plt.xlabel('Index of City')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, len(graph))
        plt.title("Cosine Simialrity Between Actual\nand Predicted Counts for All Cities")
        plt.savefig("output/RegressionCA_CosineSim.png")
        plt.clf()

            
    
def parse_args():
    parser = ca_make_arg_parser()
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainer = RegressionTrainer(args)
    data = Data.create_from_args(args)
    num_folds = len(data.partitions)
    trainer.train(data.partitions[:num_folds - 1], data.graph)
    trainer.test(data.partitions[num_folds - 1], data.graph)
