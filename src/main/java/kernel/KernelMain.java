package kernel;

import core.Evaluator;

import java.io.File;

public class KernelMain {
    public static void main(String[] args) {
        File file = new File("data/matchmaker_fixed.arff");
        String absolutePath = file.getAbsolutePath();

        Evaluator evaluator = new Evaluator(new KernelMethodClassifier("KernelMethodClassifier"), absolutePath);

        evaluator.evaluateWholeSet();
        evaluator.evaluateCV();
    }
}
