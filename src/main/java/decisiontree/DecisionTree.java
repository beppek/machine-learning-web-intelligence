package decisiontree;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class DecisionTree {

    private String path;
    private Instances data;
    private Classifier cl;

    DecisionTree(String path) throws Exception {
        this.path = path;
        readData();
    }

    public void train() throws Exception {
        cl = new J48();
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
        data.setClassIndex(3);
    }
}
