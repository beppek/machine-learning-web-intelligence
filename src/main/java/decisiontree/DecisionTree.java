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

    DecisionTree(String path) {
        this.path = path;
        readData();
    }

    public void train() {
        try {
            cl = new J48();
            cl.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void test() {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cl, data, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void readData() {
        ConverterUtils.DataSource source = null;
        try {
            source = new ConverterUtils.DataSource(path);
            data = source.getDataSet();
            data.setClassIndex(3);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
