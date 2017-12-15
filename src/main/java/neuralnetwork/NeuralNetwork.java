package neuralnetwork;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class NeuralNetwork {
    private String path;
    private Instances data;
    private Classifier cl;

    NeuralNetwork(String path) throws Exception {
        this.path = path;
        readData();
    }

    public void train() throws Exception {
        cl = new MultilayerPerceptron();
        cl.buildClassifier(data);
    }

    public void test() throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cl, data, 10, new java.util.Random(1));
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }

    private void readData() throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(8);
    }
}
