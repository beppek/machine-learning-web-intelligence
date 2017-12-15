package naivebayes;

import core.Evaluator;

import java.io.File;

public class NaiveBayesMain {
    public static void main(String[] args) {
        try {
            //Train the weka implementation on the wiki set
            NaiveBayes wnb = new NaiveBayes("data/wikipedia_70.arff");
            wnb.train();
            wnb.test();
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Now train my implementation on the fifa set
        File file = new File("data/FIFA_skill_nominal.arff");
        String absolutePath = file.getAbsolutePath();
        Evaluator evaluator = new Evaluator(new NaiveBayesClassifier(file.getName()), absolutePath);
        evaluator.evaluateWholeSet();
        evaluator.evaluateCV();
    }
}
