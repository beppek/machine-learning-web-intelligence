package svm;

import core.Evaluator;

import java.io.File;

public class SVMMain {
    public static void main(String[] args) {
        File file = new File("data/matchmaker_fixed.arff");
        String absolutePath = file.getAbsolutePath();
        Evaluator evaluator = new Evaluator(new SVMClassifier("SupportVectorMachineClassifier"), absolutePath);
        evaluator.evaluateWholeSet();
    }
}
