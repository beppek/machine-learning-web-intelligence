package naivebayes;

import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.*;

public class NaiveBayes {

    private String filename;
    private Instances data;
    private Classifier cl;

    public NaiveBayes(String filename) {
        this.filename = filename;
        readData();
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

    public void train() {
        try {
            cl = new NaiveBayesMultinomial();
            cl.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void readData() {
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
            Instances raw = source.getDataSet();

            StringToWordVector stw = new StringToWordVector(10000);
            stw.setLowerCaseTokens(true);
            stw.setInputFormat(raw);
            data = Filter.useFilter(raw, stw);
            data.setClassIndex(0);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }
}
