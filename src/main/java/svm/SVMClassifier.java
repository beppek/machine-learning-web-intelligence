package svm;

import core.Classifier;
import core.Dataset;
import core.Instance;
import core.Result;
import libsvm.*;

public class SVMClassifier implements Classifier {

    private Dataset data;
    private String name;
    private svm_problem problem;
    private svm_model model;

    SVMClassifier(String name) {
        this.name = name;
    }

    public void convertData() {
        int n = data.noInstances();
        problem = new svm_problem();
        problem.y = new double[n];
        problem.l = n;
        problem.x = new svm_node[n][data.noAttributes() -1];
        for (int i = 0; i < data.noInstances(); i++) {
            Instance inst = data.getInstance(i);
            double[] vals = inst.getAttributeArrayNumerical();
            problem.x[i] = new svm_node[data.noAttributes() - 1];

            for (int a = 0; a < data.noAttributes() - 1; a++) {
                svm_node node = new svm_node();
                node.index = a;
                node.value = vals[a];
                problem.x[i][a] = node;
            }
            problem.y[i] = inst.getClassAttribute().numericalValue();
        }
    }

    @Override
    public void train(Dataset train) {
        data = train;
        convertData();
        svm_parameter param = new svm_parameter();
        param.probability = 1;
        param.gamma = 0.5;
        param.nu = 0.5;
        param.C = 100;
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.cache_size = 20000;
        param.eps = 0.001;

        model = svm.svm_train(problem, param);
    }

    @Override
    public Result classify(Instance inst) {
        double[] vals = inst.getAttributeArrayNumerical();
        int no_classes = data.noClassValues();

        svm_node[] nodes = new svm_node[vals.length];
        for (int i =0; i < vals.length; i++) {
            svm_node node = new svm_node();
            node.index = i;
            node.value = vals[i];
            nodes[i] = node;
        }

        int[] labels = new int[no_classes];
        svm.svm_get_labels(model, labels);
        double[] prob_estimates = new double[no_classes];

        double cVal = svm.svm_predict_probability(model, nodes, prob_estimates);
        return new Result(cVal);
    }

    @Override
    public String toString() {
        return name;
    }
}
